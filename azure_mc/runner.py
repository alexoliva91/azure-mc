"""
AZURE2 execution and single Monte Carlo run logic.
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import subprocess
import uuid
from subprocess import Popen, PIPE
from typing import Optional

import numpy as np

from .models import Level
from .io import write_input_file, parse_extrap, parse_output_file, get_data_output_files
from .parameters import log_prior

log = logging.getLogger(__name__)


def generate_levels(
    levels: list[list[Level]],
    addresses: list[tuple],
    theta: np.ndarray,
) -> list[Level]:
    """Apply parameter vector to levels, return flat list of Level objects."""
    new_levels = copy.deepcopy(levels)
    for theta_i, (gi, ri, kind) in zip(theta, addresses):
        if kind == "energy":
            for sl in new_levels[gi]:
                sl.energy = theta_i
        else:
            setattr(new_levels[gi][ri], kind, theta_i)
    return [l for group in new_levels for l in group]


def run_azure2(
    input_filename: str,
    choice: int = 3,
    use_brune: bool = True,
    use_gsl: bool = True,
    ext_par_file: str = "\n",
    ext_capture_file: str = "\n",
    command: str = "AZURE2",
    timeout: int = 600,
) -> tuple[str, str, int]:
    """
    Launch AZURE2 in --no-gui mode with the given menu choice.

    choice=3 → "Extrapolate Without Data"
    """
    cl_args = [command, input_filename, "--no-gui", "--no-readline"]
    if use_brune:
        cl_args += ["--use-brune"]
    if use_gsl:
        cl_args += ["--gsl-coul"]

    options = f"{choice}\n{ext_par_file}{ext_capture_file}"
    # Force single-threaded execution so multiprocessing can
    # parallelise across walkers / samples instead.
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    p = Popen(cl_args, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env)
    try:
        stdout, stderr = p.communicate(options.encode("utf-8"), timeout=timeout)
        return stdout.decode("utf-8"), stderr.decode("utf-8"), p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return "", "TIMEOUT", -1


def run_single(
    run_id: int,
    old_contents: list[str],
    levels: list[list[Level]],
    addresses: list[tuple],
    theta: np.ndarray,
    extrap_files: list[str],
    azure2_cmd: str,
    use_brune: bool,
    use_gsl: bool,
    base_tmp_dir: str,
    keep_tmp: bool,
    timeout: int,
    norm_updates: Optional[list[tuple[int, float]]] = None,
) -> tuple[int, Optional[dict[str, np.ndarray]], str]:
    """Execute one AZURE2 extrapolation run.

    Returns
    -------
    run_id : int
    results : dict[str, np.ndarray] | None
        Mapping extrap filename → (n_pts, 3) array with columns
        [energy, cross_section, s_factor].  None on failure.
    msg : str
    """
    tag = f"run_{run_id:06d}"
    run_dir = os.path.join(base_tmp_dir, tag)
    output_dir = os.path.join(run_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    azr_path = os.path.join(run_dir, "input.azr")
    new_levels = generate_levels(levels, addresses, theta)
    write_input_file(old_contents, new_levels, azr_path, output_dir,
                     norm_updates=norm_updates)

    stdout, stderr, rc = run_azure2(
        azr_path,
        choice=3,
        use_brune=use_brune,
        use_gsl=use_gsl,
        command=azure2_cmd,
        timeout=timeout,
    )

    if rc != 0:
        msg = f"Run {run_id}: AZURE2 exit code {rc}\n{stderr[:300]}"
        if not keep_tmp:
            shutil.rmtree(run_dir, ignore_errors=True)
        return (run_id, None, msg)

    # Collect .extrap results per channel file
    per_file: dict[str, np.ndarray] = {}
    for ef in extrap_files:
        ef_path = os.path.join(output_dir, ef)
        if os.path.isfile(ef_path):
            arr = parse_extrap(ef_path)
            if arr.size > 0:
                per_file[ef] = arr

    if not per_file:
        msg = f"Run {run_id}: no .extrap data\nstdout[-200:]={stdout[-200:]}"
        if not keep_tmp:
            shutil.rmtree(run_dir, ignore_errors=True)
        return (run_id, None, msg)

    n_total = sum(a.shape[0] for a in per_file.values())
    if not keep_tmp:
        shutil.rmtree(run_dir, ignore_errors=True)
    return (run_id, per_file, f"Run {run_id}: OK ({n_total} pts, {len(per_file)} channels)")


# -------------------------------------------------------------------
# MCMC helpers – "Calculate With Data" single evaluation
# -------------------------------------------------------------------

def _gaussian_log_likelihood(output_arrays: list[np.ndarray]) -> float:
    """Compute Gaussian log-likelihood from AZURE2 ``.out`` arrays.

    For each data point the log-likelihood contribution is

    .. math::

        -\\ln(\\sqrt{2\\pi}\\,\\delta y_i)
        -\\frac{1}{2}\\left(\\frac{y_i - \\mu_i}{\\delta y_i}\\right)^2

    where

    * :math:`\\mu_i` = calculated cross section (col 3 of the .out file)
    * :math:`y_i`    = data cross section         (col 5)
    * :math:`\\delta y_i` = data uncertainty        (col 6)

    This matches (up to constant normalisation) the formulation used by
    `BRICK <https://github.com/odell/brick>`_.
    """
    lnl = 0.0
    for arr in output_arrays:
        mu = arr[:, 3]   # fit cross section
        y  = arr[:, 5]   # data cross section
        dy = arr[:, 6]   # data uncertainty
        # Guard against zero/negative uncertainties
        mask = dy > 0
        if not np.any(mask):
            continue
        mu, y, dy = mu[mask], y[mask], dy[mask]
        lnl += np.sum(
            -np.log(np.sqrt(2.0 * np.pi) * dy)
            - 0.5 * ((y - mu) / dy) ** 2
        )
    return lnl


def run_single_fit(
    run_id: str | int,
    old_contents: list[str],
    levels: list[list[Level]],
    addresses: list[tuple],
    theta: np.ndarray,
    data_output_files: list[str],
    azure2_cmd: str,
    use_brune: bool,
    use_gsl: bool,
    base_tmp_dir: str,
    timeout: int,
    norm_updates: Optional[list[tuple[int, float]]] = None,
) -> tuple[str | int, Optional[float], str]:
    """Run AZURE2 in *Calculate With Data* mode and return log-likelihood.

    The ``.out`` files are read directly (fitted cross section vs. data)
    and a Gaussian log-likelihood is computed in Python, following the
    same approach used by BRICK.

    Returns
    -------
    run_id : str | int
    lnl : float | None
        Gaussian log-likelihood.  *None* on failure.
    msg : str
    """
    tag = f"fit_{run_id}"
    run_dir = os.path.join(base_tmp_dir, tag)
    output_dir = os.path.join(run_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    azr_path = os.path.join(run_dir, "input.azr")
    new_levels = generate_levels(levels, addresses, theta)
    write_input_file(old_contents, new_levels, azr_path, output_dir,
                     norm_updates=norm_updates)

    stdout, stderr, rc = run_azure2(
        azr_path,
        choice=1,          # Calculate With Data
        use_brune=use_brune,
        use_gsl=use_gsl,
        command=azure2_cmd,
        timeout=timeout,
    )

    if rc != 0:
        msg = f"Fit {run_id}: AZURE2 exit {rc}\n{stderr[:300]}"
        shutil.rmtree(run_dir, ignore_errors=True)
        return (run_id, None, msg)

    # Read all .out files and compute log-likelihood
    output_arrays: list[np.ndarray] = []
    for of in data_output_files:
        of_path = os.path.join(output_dir, of)
        arr = parse_output_file(of_path)
        if arr is not None and arr.shape[0] > 0:
            output_arrays.append(arr)

    shutil.rmtree(run_dir, ignore_errors=True)

    if not output_arrays:
        return (run_id, None,
                f"Fit {run_id}: no .out data found")

    lnl = _gaussian_log_likelihood(output_arrays)
    n_pts = sum(a.shape[0] for a in output_arrays)
    return (run_id, lnl, f"Fit {run_id}: lnL={lnl:.4f} ({n_pts} pts)")


def log_probability(
    theta: np.ndarray,
    old_contents: list[str],
    levels: list[list[Level]],
    addresses: list[tuple],
    ranges: list[dict],
    n_level: int,
    norms: list,
    data_output_files: list[str],
    azure2_cmd: str,
    use_brune: bool,
    use_gsl: bool,
    base_tmp_dir: str,
    timeout: int,
) -> float:
    """Log-posterior for *emcee*: ``log_prior + log_likelihood``.

    The likelihood is computed by running AZURE2 in *Calculate With Data*
    mode, reading the ``.out`` files to compare fitted vs. measured cross
    sections, and evaluating a Gaussian log-likelihood (same approach as
    `BRICK <https://github.com/odell/brick>`_).
    """
    lp = log_prior(theta, ranges)
    if not np.isfinite(lp):
        return -np.inf

    theta_levels = theta[:n_level]
    norm_updates = None
    if norms:
        norm_updates = [
            (nf.index, float(theta[n_level + j]))
            for j, nf in enumerate(norms)
        ]

    run_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    _, lnl, _ = run_single_fit(
        run_id, old_contents, levels, addresses, theta_levels,
        data_output_files,
        azure2_cmd, use_brune, use_gsl, base_tmp_dir, timeout,
        norm_updates=norm_updates,
    )

    if lnl is None:
        return -np.inf

    return lp + lnl

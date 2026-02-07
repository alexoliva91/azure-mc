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
from .io import write_input_file, parse_extrap, parse_chi_squared
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
    p = Popen(cl_args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
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

def run_single_chi2(
    run_id: str | int,
    old_contents: list[str],
    levels: list[list[Level]],
    addresses: list[tuple],
    theta: np.ndarray,
    azure2_cmd: str,
    use_brune: bool,
    use_gsl: bool,
    base_tmp_dir: str,
    timeout: int,
    norm_updates: Optional[list[tuple[int, float]]] = None,
) -> tuple[str | int, Optional[float], str]:
    """Run AZURE2 in *Calculate With Data* mode and return total χ².

    Returns
    -------
    run_id : str | int
    chi2 : float | None
        Total χ² from ``chiSquared.out``.  *None* on failure.
    msg : str
    """
    tag = f"chi2_{run_id}"
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
        msg = f"Chi2 {run_id}: AZURE2 exit {rc}\n{stderr[:300]}"
        shutil.rmtree(run_dir, ignore_errors=True)
        return (run_id, None, msg)

    chi2_path = os.path.join(output_dir, "chiSquared.out")
    chi2 = parse_chi_squared(chi2_path)
    shutil.rmtree(run_dir, ignore_errors=True)

    if chi2 is None:
        return (run_id, None,
                f"Chi2 {run_id}: could not parse chiSquared.out")
    return (run_id, chi2, f"Chi2 {run_id}: chi2={chi2:.4f}")


def log_probability(
    theta: np.ndarray,
    old_contents: list[str],
    levels: list[list[Level]],
    addresses: list[tuple],
    ranges: list[dict],
    n_level: int,
    norms: list,
    azure2_cmd: str,
    use_brune: bool,
    use_gsl: bool,
    base_tmp_dir: str,
    timeout: int,
) -> float:
    """Log-posterior for *emcee*: ``log_prior + log_likelihood``.

    The likelihood is ``-0.5 × χ²_total`` where χ² comes from
    AZURE2's *Calculate With Data* mode.
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
    _, chi2, _ = run_single_chi2(
        run_id, old_contents, levels, addresses, theta_levels,
        azure2_cmd, use_brune, use_gsl, base_tmp_dir, timeout,
        norm_updates=norm_updates,
    )

    if chi2 is None:
        return -np.inf

    return lp - 0.5 * chi2

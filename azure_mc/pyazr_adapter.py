"""
Adapter layer bridging azure_mc's Monte Carlo workflow to the AZURE2 dev
branch's socket-based Python API (``pyazr``).

``pyazr`` is not pip-installable — it ships as a plain package directory
inside the AZURE2 dev-branch source tree (``<repo>/pyazr``). It must be made
importable either by adding that directory's *parent* to ``PYTHONPATH``
yourself, or by pointing the ``pyazr_path`` setup-YAML key (or the
corresponding CLI flag) at the ``pyazr`` package directory — this module
inserts its parent onto ``sys.path`` before importing it.

Everything here operates in AZURE2's *physical* parameter space (the same
values shown in the GUI / stored in ``<levels>``), matching azure_mc's
existing Monte Carlo semantics — the reduced-width-amplitude ("rwa") API
(``calculate_rwa``, ``params_rwa``) used by pyazr's own fitters is
intentionally not used here.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class PyazrUnavailableError(RuntimeError):
    """Raised when the ``pyazr`` package cannot be imported."""


def _import_pyazr(pyazr_path: Optional[str] = None):
    """Import and return the ``pyazr`` module, adding ``pyazr_path`` to
    ``sys.path`` first if given.

    ``pyazr_path`` should point at the ``pyazr`` package directory itself
    (e.g. ``~/repos/AZURE2-rdeboer1/pyazr``); its *parent* directory is what
    gets added to ``sys.path`` so ``import pyazr`` resolves to that package.
    """
    if pyazr_path:
        pyazr_path = os.path.abspath(os.path.expanduser(pyazr_path))
        parent = os.path.dirname(pyazr_path)
        if parent not in sys.path:
            sys.path.insert(0, parent)
    try:
        import pyazr  # noqa: F401  (imported for side effect / re-export)
    except ImportError as exc:
        raise PyazrUnavailableError(
            "Could not import 'pyazr'. Either build AZURE2 from the dev "
            "branch and add its 'pyazr' directory's parent to PYTHONPATH, "
            "or set 'pyazr_path' in the setup YAML (or pass --pyazr-path) "
            "to that directory."
        ) from exc
    return pyazr


def open_session(
    azr_filepath: str,
    nprocs: int = 1,
    binary: Optional[str] = None,
    cwd: Optional[str] = None,
    timeout: float = 1800.0,
    pyazr_path: Optional[str] = None,
):
    """Launch a persistent AZURE2 API session for the whole MC run.

    Raises :class:`PyazrUnavailableError` if ``pyazr`` cannot be imported.
    The caller owns the returned session and must call ``.close()`` (or use
    it as a context manager) when done.
    """
    pyazr = _import_pyazr(pyazr_path)
    if binary:
        binary = os.path.abspath(os.path.expanduser(binary))
    azr = pyazr.azure2(
        azr_filepath, nprocs=nprocs, binary=binary, cwd=cwd, timeout=timeout,
    )
    azr.extrap_mode()
    return azr


# ---------------------------------------------------------------------
# Free-parameter discovery
# ---------------------------------------------------------------------

@dataclass
class FreeParam:
    """One free (non-fixed) AZURE2 parameter, addressable for MC sampling."""

    index: int          # position in the full physical vector (azr.params)
    kind: str           # 'energy' | 'width' | 'norm' | 'shift'
    key: str            # short unique key, used as the mc_params.yaml key
    description: str
    nominal: float


def _fmt_spin(j) -> str:
    if j is None:
        return "?"
    return str(int(j)) if float(j).is_integer() else str(j)


def _fmt_pi(parity) -> str:
    return "+" if parity is not None and parity > 0 else "-"


def list_free_parameters(azr) -> list[FreeParam]:
    """Every non-fixed parameter of a live pyazr session.

    Keys are built from pyazr's own parameter metadata (J/parity/level for
    R-matrix parameters, pair/L/S for channels, segment for norms/shifts)
    and are NOT expected to match keys produced by azure_mc's legacy .azr
    parsing — switching a model between ``engine: legacy`` and
    ``engine: pyazr`` requires re-running ``populate``.
    """
    out: list[FreeParam] = []
    for p in azr.parameters:
        if p.fixed:
            continue
        if p.kind == "energy":
            jpi = f"{_fmt_spin(p.J)}{_fmt_pi(p.parity)}"
            key = f"J{jpi}_lvl{p.level}_energy"
            description = f"J={jpi}  energy  level={p.level}"
        elif p.kind == "width":
            jpi = f"{_fmt_spin(p.J)}{_fmt_pi(p.parity)}"
            key = f"J{jpi}_lvl{p.level}_width_pair{p.pair}_L{p.L}_S{p.S}"
            description = (
                f"J={jpi}  width  level={p.level}  pair={p.pair}  "
                f"L={p.L}  S={p.S}  rad={p.radiation_type or 'P'}"
            )
        elif p.kind == "norm":
            key = f"norm_seg{p.segment_key}"
            description = f"Normalisation factor for data segment {p.segment_key}"
        elif p.kind == "shift":
            key = f"shift_seg{p.segment_key}"
            description = f"Energy shift for data segment {p.segment_key}"
        else:
            key = f"param_idx{p.index}"
            description = f"Unclassified parameter '{p.name}'"
        out.append(FreeParam(
            index=p.index, kind=p.kind, key=key, description=description,
            nominal=float(p.value),
        ))
    return out


# ---------------------------------------------------------------------
# Channel naming (mirrors azure_mc.io.get_extrap_output_files)
# ---------------------------------------------------------------------

def channel_name_for_segment(segment) -> str:
    """Legacy-compatible channel name for a pyazr ``TestSegment``.

    Matches the ``.extrap`` output-file naming
    ``azure_mc.io.get_extrap_output_files`` derives from the same
    ``<segmentsTest>`` block, so downstream aggregation code (grouping,
    quantiles, output file naming) is unaffected by which engine produced
    the per-sample arrays.
    """
    if segment.exit_key == -1:
        return f"AZUREOut_aa={segment.entrance_key}_TOTAL_CAPTURE.extrap"
    return f"AZUREOut_aa={segment.entrance_key}_R={segment.exit_key}.extrap"


def _active_test_segments(azr):
    """Active ``<segmentsTest>`` entries, in the order pyazr's
    ``calculate*`` methods iterate calculated segments."""
    return [seg for seg in azr.extrapolations if seg.active]


# ---------------------------------------------------------------------
# Sampling + evaluation
# ---------------------------------------------------------------------

def evaluate_sample(azr, free_params: list[FreeParam], theta: np.ndarray,
                     proc: int = 0) -> dict[str, np.ndarray]:
    """Apply one sampled physical parameter vector and calculate.

    Returns ``{channel_name: array}`` with one ``(n_pts, 3)`` array per
    channel — columns ``[energy, cross_section, s_factor]`` — matching the
    shape ``azure_mc.runner.run_single`` returns for the legacy engine.
    Multiple ``<segmentsTest>`` grids that map to the same legacy channel
    name (e.g. several energy ranges of the same reaction) are concatenated
    and sorted by energy, mirroring how the legacy ``.extrap`` files
    aggregate all segments of a channel into one file.
    """
    x = np.array(azr.params, dtype=float)
    for fp, value in zip(free_params, theta):
        x[fp.index] = float(value)

    energies = azr.calculate_energies(x, proc=proc)
    cross = azr.calculate(x, proc=proc)
    sfactor = azr.calculate_sfactor(x, proc=proc)

    segments = _active_test_segments(azr)
    n = len(segments)
    if not (len(energies) == len(cross) == len(sfactor) == n):
        raise RuntimeError(
            f"pyazr returned {len(energies)}/{len(cross)}/{len(sfactor)} "
            f"calculated segments but the model declares {n} active "
            f"extrapolation segments."
        )

    by_channel: dict[str, list[np.ndarray]] = {}
    for seg, e, xs, sf in zip(segments, energies, cross, sfactor):
        name = channel_name_for_segment(seg)
        arr = np.column_stack([np.asarray(e), np.asarray(xs), np.asarray(sf)])
        by_channel.setdefault(name, []).append(arr)

    out: dict[str, np.ndarray] = {}
    for name, arrs in by_channel.items():
        combined = np.concatenate(arrs, axis=0)
        out[name] = combined[np.argsort(combined[:, 0])]
    return out

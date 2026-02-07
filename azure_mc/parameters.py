"""
Parameter discovery, value extraction, and Monte Carlo sampling.
"""

from __future__ import annotations

import numpy as np

from .constants import DATA_NORM_FACTOR_INDEX, DATA_VARY_NORM_INDEX
from .models import Level, Parameter, NormFactor
from .io import read_levels, read_data_segments


def discover_free_parameters(
    contents: list[str],
) -> tuple[list[Parameter], list[NormFactor], list[tuple]]:
    """
    Walk the levels and data segments to find free parameters.

    Returns
    -------
    parameters : list[Parameter]
        Free level parameters (energies + widths).
    norm_factors : list[NormFactor]
        Free normalisation factors.
    addresses : list[tuple]
        (group_index, row_index_in_group, kind) for each Parameter.
    """
    levels = read_levels(contents)
    parameters: list[Parameter] = []
    addresses: list[tuple] = []
    jpis: list[float] = []

    for gi, group in enumerate(levels):
        first = group[0]
        jpi = (first.spin, first.parity)
        jpis.append(jpi)
        rank = jpis.count(jpi)

        for i, sublevel in enumerate(group):
            if sublevel.include:
                if i == 0 and not sublevel.energy_fixed:
                    parameters.append(
                        Parameter(
                            sublevel.spin,
                            sublevel.parity,
                            "energy",
                            i + 1,
                            rank=rank,
                        )
                    )
                    addresses.append((gi, i, "energy"))
                if not sublevel.width_fixed:
                    is_anc = sublevel.energy < sublevel.separation_energy
                    parameters.append(
                        Parameter(
                            sublevel.spin,
                            sublevel.parity,
                            "width",
                            i + 1,
                            rank=rank,
                            is_anc=is_anc,
                        )
                    )
                    addresses.append((gi, i, "width"))

    # Normalization factors
    norm_factors: list[NormFactor] = []
    data_segs = read_data_segments(contents)
    for idx, seg in enumerate(data_segs):
        if len(seg) > DATA_VARY_NORM_INDEX:
            vary = int(seg[DATA_VARY_NORM_INDEX])
            if vary:
                norm_factors.append(NormFactor(idx))

    return parameters, norm_factors, addresses


def get_input_values(
    contents: list[str],
    parameters: list[Parameter],
    norm_factors: list[NormFactor],
    addresses: list[tuple],
) -> list[float]:
    """Read current values of free parameters from the input file."""
    levels = read_levels(contents)
    values = []
    for gi, ri, kind in addresses:
        values.append(getattr(levels[gi][ri], kind))
    data_segs = read_data_segments(contents)
    for nf in norm_factors:
        values.append(float(data_segs[nf.index][DATA_NORM_FACTOR_INDEX]))
    return values


def sample_theta(
    nominals: np.ndarray,
    ranges: list[dict],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one MC sample for every free parameter."""
    theta = np.empty(len(nominals))
    for i, (nom, r) in enumerate(zip(nominals, ranges)):
        dist = r.get("distribution", "uniform")
        lo = r.get("low", nom * 0.8 if nom >= 0 else nom * 1.2)
        hi = r.get("high", nom * 1.2 if nom >= 0 else nom * 0.8)
        if lo > hi:
            lo, hi = hi, lo
        if lo == hi:
            lo = nom - max(abs(nom) * 0.2, 1.0)
            hi = nom + max(abs(nom) * 0.2, 1.0)

        if dist == "gaussian":
            sigma = r.get("sigma", (hi - lo) / 4.0 if hi != lo else 1.0)
            theta[i] = rng.normal(nom, sigma)
        else:
            theta[i] = rng.uniform(lo, hi)
    return theta


# -------------------------------------------------------------------
# MCMC helpers
# -------------------------------------------------------------------

def log_prior(
    theta: np.ndarray,
    ranges: list[dict],
) -> float:
    """Compute the log-prior probability.

    * **uniform** prior → flat within ``[low, high]``, ``-inf`` outside.
    * **gaussian** prior → :math:`\\mathcal{N}(\\text{nominal}, \\sigma)`
      with hard bounds at ``[low, high]``.
    """
    lp = 0.0
    for val, r in zip(theta, ranges):
        dist = r.get("distribution", "uniform")
        lo = r.get("low", -np.inf)
        hi = r.get("high", np.inf)

        # Hard bounds always enforced
        if val < lo or val > hi:
            return -np.inf

        if dist == "gaussian":
            nom = r.get("nominal", (lo + hi) / 2.0)
            sigma = r.get("sigma", (hi - lo) / 4.0 if hi != lo else 1.0)
            if sigma <= 0:
                sigma = 1.0
            lp += -0.5 * ((val - nom) / sigma) ** 2
        # uniform → constant log-prior (contributes 0)
    return lp


def initialize_walkers(
    nominals: np.ndarray,
    ranges: list[dict],
    n_walkers: int,
    rng: np.random.Generator,
    spread: float = 1e-4,
) -> np.ndarray:
    """Create initial walker positions as a tight ball around *nominals*.

    Parameters
    ----------
    spread : float
        Fraction of the parameter range used as standard deviation for the
        initial perturbation.
    """
    ndim = len(nominals)
    p0 = np.empty((n_walkers, ndim))

    for i, (nom, r) in enumerate(zip(nominals, ranges)):
        lo = r.get("low", nom - 1.0)
        hi = r.get("high", nom + 1.0)
        width = (hi - lo) * spread
        if width == 0:
            width = max(abs(nom) * spread, 1e-10)

        for w in range(n_walkers):
            val = nom + width * rng.standard_normal()
            val = max(lo, min(hi, val))        # clamp within bounds
            p0[w, i] = val

    return p0

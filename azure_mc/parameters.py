"""
Parameter discovery, value extraction, and Monte Carlo sampling.
"""

from __future__ import annotations

import logging
import numpy as np

from .constants import DATA_NORM_FACTOR_INDEX, DATA_VARY_NORM_INDEX
from .models import Level, Parameter, NormFactor
from .io import read_levels, read_data_segments

log = logging.getLogger(__name__)

# ---- scipy lazy import ---------------------------------------------------

_scipy_stats = None  # populated on first use


def _get_scipy_stats():
    """Import ``scipy.stats`` lazily so scipy is only required when needed."""
    global _scipy_stats
    if _scipy_stats is None:
        try:
            from scipy import stats as _st
            _scipy_stats = _st
        except ImportError:
            raise ImportError(
                "scipy is required for distributions other than "
                "'uniform' and 'gaussian'.  Install with:  pip install scipy"
            )
    return _scipy_stats


# Keys that belong to the range configuration, not to scipy dist_params.
_RESERVED_KEYS = frozenset({
    "distribution", "low", "high", "nominal", "sigma", "mu",
    "description", "dist_params",
})

# Built-in distribution names handled without scipy.
_BUILTIN_DISTRIBUTIONS = frozenset({
    "uniform", "gaussian", "normal", "lognormal",
})


def _resolve_bounds(nom: float, r: dict) -> tuple[float, float]:
    """Return (lo, hi) from a range dict.

    For **uniform** distributions ``low``/``high`` are required so
    sensible ±20 % defaults are filled in.  For every other distribution
    the default is ``(-inf, +inf)`` — i.e. no clipping unless the user
    explicitly sets bounds.
    """
    dist = r.get("distribution", "uniform")
    if dist == "uniform":
        lo = r.get("low", nom * 0.8 if nom >= 0 else nom * 1.2)
        hi = r.get("high", nom * 1.2 if nom >= 0 else nom * 0.8)
        if lo > hi:
            lo, hi = hi, lo
        if lo == hi:
            lo = nom - max(abs(nom) * 0.2, 1.0)
            hi = nom + max(abs(nom) * 0.2, 1.0)
    else:
        lo = r.get("low", -np.inf)
        hi = r.get("high", np.inf)
    return lo, hi


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
    """Draw one MC sample for every free parameter.

    Supported distributions
    -----------------------
    **uniform**
        Flat between ``low`` and ``high``.
    **gaussian** / **normal**
        Normal centred on ``nominal`` with std ``sigma``.
    **lognormal**
        Log-normal parameterised by ``mu`` and ``sigma`` of ``ln(X)``.
        Defaults: ``mu = ln(|nominal|)``, ``sigma = 1``.
    *<any scipy.stats name>*
        Looked up in ``scipy.stats``; shape / loc / scale parameters are
        passed via the ``dist_params`` dict.

    If ``low`` / ``high`` are set they act as hard clipping bounds;
    otherwise no clipping is applied.
    """
    theta = np.empty(len(nominals))
    for i, (nom, r) in enumerate(zip(nominals, ranges)):
        dist = r.get("distribution", "uniform")
        lo, hi = _resolve_bounds(nom, r)

        if dist == "uniform":
            theta[i] = rng.uniform(lo, hi)

        elif dist in ("gaussian", "normal"):
            sigma = r.get("sigma",
                          (hi - lo) / 4.0
                          if np.isfinite(hi) and np.isfinite(lo) and hi != lo
                          else 1.0)
            theta[i] = rng.normal(nom, sigma)

        elif dist == "lognormal":
            mu = r.get("mu", np.log(abs(nom)) if nom != 0 else 0.0)
            sigma = r.get("sigma", 1.0)
            theta[i] = rng.lognormal(mu, sigma)

        else:
            # Generic scipy.stats distribution
            stats = _get_scipy_stats()
            sp_cls = getattr(stats, dist, None)
            if sp_cls is None:
                raise ValueError(
                    f"Unknown distribution '{dist}'. Must be 'uniform', "
                    "'gaussian', 'lognormal', or a scipy.stats distribution "
                    f"name (see scipy.stats docs).  Available: "
                    f"https://docs.scipy.org/doc/scipy/reference/stats.html"
                )
            dist_params = dict(r.get("dist_params", {}))
            # Use scipy's rvs with the numpy Generator
            theta[i] = sp_cls.rvs(**dist_params, random_state=rng)

        # Enforce hard bounds (only if explicitly set)
        if np.isfinite(lo) or np.isfinite(hi):
            theta[i] = np.clip(theta[i], lo, hi)

    return theta


# -------------------------------------------------------------------
# MCMC helpers
# -------------------------------------------------------------------

def log_prior(
    theta: np.ndarray,
    ranges: list[dict],
) -> float:
    r"""Compute the log-prior probability.

    * **uniform** → flat within ``[low, high]``, :math:`-\infty` outside.
    * **gaussian** / **normal** →
      :math:`\mathcal{N}(\text{nominal}, \sigma)` with hard bounds.
    * **lognormal** →
      :math:`\text{LogNormal}(\mu, \sigma)` with hard bounds.
      Defaults: ``mu = \ln(|\text{nominal}|)``, ``sigma = 1``.
    * *<scipy.stats name>* → ``logpdf`` from the frozen scipy
      distribution, with hard bounds at ``[low, high]``.
    """
    lp = 0.0
    for val, r in zip(theta, ranges):
        dist = r.get("distribution", "uniform")
        lo = r.get("low", -np.inf)
        hi = r.get("high", np.inf)

        # Hard bounds always enforced
        if val < lo or val > hi:
            return -np.inf

        if dist == "uniform":
            pass  # flat prior → contributes 0

        elif dist in ("gaussian", "normal"):
            nom = r.get("nominal", (lo + hi) / 2.0)
            sigma = r.get("sigma", (hi - lo) / 4.0 if hi != lo else 1.0)
            if sigma <= 0:
                sigma = 1.0
            lp += -0.5 * ((val - nom) / sigma) ** 2

        elif dist == "lognormal":
            if val <= 0:
                return -np.inf
            nom_val = r.get("nominal", 1.0)
            mu = r.get("mu", np.log(abs(nom_val)) if nom_val != 0 else 0.0)
            sigma = r.get("sigma", 1.0)
            if sigma <= 0:
                sigma = 1.0
            lp += -(np.log(val) - mu) ** 2 / (2 * sigma ** 2) - np.log(val * sigma)

        else:
            # Generic scipy.stats distribution
            stats = _get_scipy_stats()
            sp_cls = getattr(stats, dist, None)
            if sp_cls is None:
                raise ValueError(
                    f"Unknown distribution '{dist}' for log-prior.  "
                    "Must be 'uniform', 'gaussian', 'lognormal', or a "
                    "scipy.stats distribution name."
                )
            dist_params = dict(r.get("dist_params", {}))
            logp = sp_cls.logpdf(val, **dist_params)
            if not np.isfinite(logp):
                return -np.inf
            lp += logp

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
        lo = r.get("low", -np.inf)
        hi = r.get("high", np.inf)
        if np.isfinite(lo) and np.isfinite(hi):
            width = (hi - lo) * spread
        else:
            width = max(abs(nom) * spread, 1e-10)
        if width == 0:
            width = max(abs(nom) * spread, 1e-10)

        for w in range(n_walkers):
            val = nom + width * rng.standard_normal()
            if np.isfinite(lo):
                val = max(lo, val)
            if np.isfinite(hi):
                val = min(hi, val)
            p0[w, i] = val

    return p0

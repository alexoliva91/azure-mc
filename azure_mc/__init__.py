"""
AZURE2 Monte Carlo uncertainty propagation tool.

Parses an AZURE2 .azr input file, identifies free (variable) parameters,
samples them via Monte Carlo within
user-defined ranges, runs AZURE2 in "Extrapolate Without Data" mode in
parallel, and collects the resulting cross sections or S-factors.
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

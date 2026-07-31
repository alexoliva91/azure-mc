"""
Column indices for .azr file parsing.

Verified against the AZURE2 dev-branch C++ headers (NucLine.h / SegLine.h /
ExtrapLine.h) as of 2026-07-31 -- see azure_mc/io.py's data_norm_indices()
for the one field group whose position isn't fixed.
"""

# Level row indices (<levels> / NucLine.h -- 31 fields: levelJ, levelPi,
# levelE, levelFix, aa, ir, s2, l2, levelID, isActive, channelFix, gamma,
# j1, pi1, j2, pi2, e2, m1, m2, z1, z2, entranceSepE, sepE, j3, pi3, e3,
# pType, chRad, g1, g2, ecMultMask)
J_INDEX = 0
PI_INDEX = 1
ENERGY_INDEX = 2
ENERGY_FIXED_INDEX = 3
CHANNEL_INDEX = 5
LEVEL_INCLUDE_INDEX = 9
WIDTH_FIXED_INDEX = 10
WIDTH_INDEX = 11
SEPARATION_ENERGY_INDEX = 22
CHANNEL_RADIUS_INDEX = 27

# Data segment indices (<segmentsData> / SegLine.h). The isActive/
# entranceKey/exitKey/.../isDiff prefix (indices 0-7) is always fixed, but
# a phase-shift segment (isDiff == 7th field == 2) carries two extra fields
# (phaseJ, phaseL) right after it, shifting dataNorm/varyNorm/dataNormError
# (and everything after, including the data file path) two columns later.
# Use data_norm_indices() in io.py to get the correct per-row offset rather
# than assuming these fixed positions for every segment.
DATA_INCLUDE_INDEX = 0
DATA_IN_CHANNEL_INDEX = 1
DATA_OUT_CHANNEL_INDEX = 2
DATA_ISDIFF_INDEX = 7
DATA_NORM_FACTOR_INDEX = 8   # only correct when isDiff != 2; see data_norm_indices()
DATA_VARY_NORM_INDEX = 9     # only correct when isDiff != 2; see data_norm_indices()
# Common-case position of the data file path (isDiff != 2, with the optional
# energyShift/energyShiftError/varyEnergyShift triple present). Not fixed in
# general -- unused by azure_mc today, kept for reference/documentation only.
DATA_FILEPATH_INDEX = 14

# Config section indices
OUTPUT_DIR_INDEX = 1  # line index inside <config>

"""
I/O helpers for reading / writing .azr files and parsing AZURE2 output.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np

from .constants import (
    J_INDEX,
    PI_INDEX,
    ENERGY_INDEX,
    WIDTH_INDEX,
    CHANNEL_RADIUS_INDEX,
    DATA_INCLUDE_INDEX,
    DATA_IN_CHANNEL_INDEX,
    DATA_OUT_CHANNEL_INDEX,
    DATA_NORM_FACTOR_INDEX,
    DATA_FILEPATH_INDEX,
    OUTPUT_DIR_INDEX,
)
from .models import Level


# -------------------------------------------------------------------
# Read helpers
# -------------------------------------------------------------------

def read_input_file(filename: str) -> list[str]:
    with open(filename, "r") as f:
        return f.read().split("\n")


def read_level_contents(contents: list[str]) -> list[str]:
    start = contents.index("<levels>") + 1
    stop = contents.index("</levels>")
    return contents[start:stop]


def read_levels(contents: list[str]) -> list[list[Level]]:
    """Return list of level groups (each group = list of Level rows)."""
    level_contents = read_level_contents(contents)
    levels: list[list[Level]] = []
    sublevels: list[Level] = []
    for row in level_contents:
        if row.strip() != "":
            sublevels.append(Level(row))
        else:
            if sublevels:
                levels.append(sublevels)
                sublevels = []
    if sublevels:
        levels.append(sublevels)
    return levels


def read_data_segments(contents: list[str]):
    """Return list of parsed data-segment rows."""
    start = contents.index("<segmentsData>") + 1
    stop = contents.index("</segmentsData>")
    segments = []
    for row in contents[start:stop]:
        if row.strip():
            segments.append(row.split())
    return segments


def read_test_segments(contents: list[str]):
    """Return parsed test (no-data) segment rows."""
    start = contents.index("<segmentsTest>") + 1
    stop = contents.index("</segmentsTest>")
    segments = []
    for row in contents[start:stop]:
        if row.strip():
            segments.append(row.split())
    return segments


def get_extrap_output_files(contents: list[str]) -> list[str]:
    """Derive .extrap output filenames from <segmentsTest>."""
    test_segs = read_test_segments(contents)
    files = set()
    for seg in test_segs:
        include = int(seg[0])
        if include:
            in_ch = seg[1]
            out_ch = seg[2]
            if int(out_ch) == -1:
                fname = f"AZUREOut_aa={in_ch}_TOTAL_CAPTURE.extrap"
            else:
                fname = f"AZUREOut_aa={in_ch}_R={out_ch}.extrap"
            files.add(fname)
    return sorted(files)


# -------------------------------------------------------------------
# Write helpers
# -------------------------------------------------------------------

def _replace_token(parts: list[str], token_index: int, new_value: str):
    """Replace a token in a ``re.split(r'(\\s+)', ...)`` list, preserving alignment."""
    pos = 2 * token_index
    if pos >= len(parts):
        return
    old_val = parts[pos]
    if len(new_value) <= len(old_val):
        parts[pos] = new_value.rjust(len(old_val))
    else:
        overflow = len(new_value) - len(old_val)
        parts[pos] = new_value
        if pos > 0:
            ws = parts[pos - 1]
            parts[pos - 1] = ws[:max(1, len(ws) - overflow)]


def write_input_file(
    old_contents: list[str],
    new_levels: list[Level],
    output_filename: str,
    output_dir: str,
    norm_updates: Optional[list[tuple[int, float]]] = None,
):
    """
    Write a new .azr file with modified levels and output directory.
    Preserves original formatting / whitespace outside the modified columns.

    Parameters
    ----------
    norm_updates : list of (segment_index, new_norm_value), optional
        Normalization factor overrides for ``<segmentsData>``.
    """
    contents = list(old_contents)

    # Apply norm factor updates to segmentsData
    if norm_updates:
        seg_start = contents.index("<segmentsData>") + 1
        seg_stop = contents.index("</segmentsData>")
        norm_dict = dict(norm_updates)
        data_row_idx = 0
        for line_idx in range(seg_start, seg_stop):
            if contents[line_idx].strip():
                if data_row_idx in norm_dict:
                    line = contents[line_idx]
                    leading = line[:len(line) - len(line.lstrip())]
                    parts = re.split(r'(\s+)', line.lstrip())
                    _replace_token(parts, DATA_NORM_FACTOR_INDEX,
                                   str(norm_dict[data_row_idx]))
                    contents[line_idx] = leading + ''.join(parts)
                data_row_idx += 1

    start = contents.index("<levels>") + 1
    stop = contents.index("</levels>")
    old_levels_raw = contents[start:stop]
    nlines = len(old_levels_raw)
    level_indices = [i for i, line in enumerate(old_levels_raw) if line.strip() != ""]
    blank_indices = set(i for i in range(nlines) if i not in level_indices)
    assert len(level_indices) == len(new_levels), (
        f"Level count mismatch: {len(level_indices)} vs {len(new_levels)}"
    )

    new_level_data = []
    j = 0
    for i in range(nlines):
        if i in blank_indices:
            new_level_data.append("")
        else:
            level = new_levels[j]
            line = old_levels_raw[i]
            leading = line[:len(line) - len(line.lstrip())]
            parts = re.split(r'(\s+)', line.lstrip())
            spin_str = (str(int(level.spin))
                        if level.spin == int(level.spin)
                        else str(level.spin))
            _replace_token(parts, J_INDEX, spin_str)
            _replace_token(parts, PI_INDEX, str(level.parity))
            _replace_token(parts, ENERGY_INDEX, str(level.energy))
            _replace_token(parts, WIDTH_INDEX, str(level.width))
            _replace_token(parts, CHANNEL_RADIUS_INDEX, str(level.channel_radius))
            new_level_data.append(leading + ''.join(parts))
            j += 1

    with open(output_filename, "w") as f:
        f.write(contents[0] + "\n")
        f.write(contents[1] + "\n")
        f.write(output_dir + "/\n")
        for row in contents[OUTPUT_DIR_INDEX + 1 : start]:
            f.write(row + "\n")
        for row in new_level_data:
            f.write(row + "\n")
        f.write("</levels>\n")
        for row in contents[stop + 1 :]:
            f.write(row + "\n")


# -------------------------------------------------------------------
# Parse AZURE2 .extrap output
# -------------------------------------------------------------------

def parse_extrap(filepath: str) -> np.ndarray:
    """
    Parse an AZURE2 .extrap output file.

    Columns in .extrap:
        0 - CoM Energy (MeV)
        1 - Excitation Energy (MeV)
        2 - CoM Angle (degrees)
        3 - Cross Section (barns)
        4 - S-factor (MeV b)

    Returns
    -------
    np.ndarray, shape (n_pts, 3)
        Columns: [energy, cross_section, s_factor]
    """
    data = []
    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    data.append((
                        float(parts[0]),   # energy
                        float(parts[3]),   # cross section
                        float(parts[4]),   # s-factor
                    ))
                except ValueError:
                    continue
    return np.array(data) if data else np.empty((0, 3))


# -------------------------------------------------------------------
# Parse AZURE2 chiSquared.out
# -------------------------------------------------------------------

def parse_chi_squared(filepath: str) -> Optional[float]:
    """Read total chi-squared from AZURE2's ``chiSquared.out``.

    The file lists χ²/N per segment; the **last** non-empty line
    contains the total χ² of the calculation.
    """
    if not os.path.isfile(filepath):
        return None
    last_value: Optional[float] = None
    with open(filepath, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parts = stripped.split()
                last_value = float(parts[-1])
            except (ValueError, IndexError):
                continue
    return last_value


def get_data_output_files(contents: list[str]) -> list[str]:
    """Derive ``.out`` output filenames from ``<segmentsData>``."""
    data_segs = read_data_segments(contents)
    files: set[str] = set()
    for seg in data_segs:
        include = int(seg[DATA_INCLUDE_INDEX])
        if include:
            in_ch = seg[DATA_IN_CHANNEL_INDEX]
            out_ch = seg[DATA_OUT_CHANNEL_INDEX]
            if int(out_ch) == -1:
                fname = f"AZUREOut_aa={in_ch}_TOTAL_CAPTURE.out"
            else:
                fname = f"AZUREOut_aa={in_ch}_R={out_ch}.out"
            files.add(fname)
    return sorted(files)


def resolve_data_paths(contents: list[str], azr_base_dir: str) -> list[str]:
    """Resolve relative data-file paths in ``<segmentsData>`` to absolute.

    This is needed when writing ``.azr`` files to temporary directories
    (e.g. for MCMC evaluations) so that data files are still found.
    """
    contents = list(contents)               # shallow copy
    try:
        start = contents.index("<segmentsData>") + 1
        stop = contents.index("</segmentsData>")
    except ValueError:
        return contents

    for i in range(start, stop):
        line = contents[i]
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) > DATA_FILEPATH_INDEX:
            filepath = parts[DATA_FILEPATH_INDEX]
            if not os.path.isabs(filepath):
                abs_path = os.path.normpath(
                    os.path.join(azr_base_dir, filepath)
                )
                # Replace the filepath token (always the last token)
                contents[i] = line[: line.rfind(filepath)] + abs_path
    return contents

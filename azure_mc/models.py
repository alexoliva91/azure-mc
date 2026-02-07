"""
Data model classes: Level, Parameter, NormFactor.
"""

from __future__ import annotations

from .constants import (
    J_INDEX,
    PI_INDEX,
    ENERGY_INDEX,
    ENERGY_FIXED_INDEX,
    CHANNEL_INDEX,
    LEVEL_INCLUDE_INDEX,
    WIDTH_FIXED_INDEX,
    WIDTH_INDEX,
    SEPARATION_ENERGY_INDEX,
    CHANNEL_RADIUS_INDEX,
)


class Level:
    """One row inside the ``<levels>`` section of an .azr file."""

    def __init__(self, row_str: str):
        row = row_str.split()
        self.spin = float(row[J_INDEX])
        self.parity = int(row[PI_INDEX])
        self.energy = float(row[ENERGY_INDEX])
        self.energy_fixed = int(row[ENERGY_FIXED_INDEX])
        self.width = float(row[WIDTH_INDEX])
        self.width_fixed = int(int(row[WIDTH_FIXED_INDEX]) or self.width == 0)
        self.channel_radius = float(row[CHANNEL_RADIUS_INDEX])
        self.channel = int(row[CHANNEL_INDEX])
        self.separation_energy = float(row[SEPARATION_ENERGY_INDEX])
        self.include = int(row[LEVEL_INCLUDE_INDEX])


class Parameter:
    """A sampled (free) parameter."""

    def __init__(self, spin, parity, kind, channel, rank=1, is_anc=False):
        self.spin = spin
        self.parity = parity
        self.kind = kind          # "energy" or "width"
        self.channel = int(channel)
        self.rank = rank
        self.is_anc = is_anc

        jpi_label = "+" if self.parity == 1 else "-"
        subscript = f"{rank:d},{channel:d}"
        superscript = f"({jpi_label}{spin:.1f})"
        if kind == "energy":
            self.label = f"E_{subscript}^{superscript}"
        elif is_anc:
            self.label = f"C_{subscript}^{superscript}"
        else:
            self.label = f"G_{subscript}^{superscript}"

    def key(self) -> str:
        """Short unique key used in the YAML config."""
        pi = "+" if self.parity == 1 else "-"
        return f"J{self.spin}{pi}_rank{self.rank}_{self.kind}_ch{self.channel}"

    def description(self) -> str:
        pi = "+" if self.parity == 1 else "-"
        kind_label = self.kind
        if self.kind == "width" and self.is_anc:
            kind_label = "ANC"
        return (
            f"J={self.spin}{pi}  {kind_label}  rank={self.rank}  "
            f"channel={self.channel}  [{self.label}]"
        )


class NormFactor:
    """A sampled normalisation factor from a data segment."""

    def __init__(self, dataset_index):
        self.index = dataset_index
        self.label = f"n_{self.index + 1}"

    def key(self) -> str:
        return f"norm_{self.index}"

    def description(self) -> str:
        return f"Normalisation factor for data segment {self.index}"

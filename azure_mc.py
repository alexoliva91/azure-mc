#!/usr/bin/env python3
"""
AZURE2 Monte Carlo uncertainty propagation tool.

This script is a thin wrapper around the ``azure_mc`` package.
All logic lives in azure_mc/ — this file is kept for backwards
compatibility so that ``python azure_mc.py <command> ...`` still works.

Preferred invocation:  python -m azure_mc <command> ...
"""

from azure_mc.cli import main

if __name__ == "__main__":
    main()

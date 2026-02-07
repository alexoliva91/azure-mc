"""
CLI entry point — argparse setup and dispatch.
"""

from __future__ import annotations

import argparse

from .commands import cmd_populate, cmd_run, cmd_summary, cmd_recompute_quantiles


def main():
    parser = argparse.ArgumentParser(
        description="AZURE2 Monte Carlo uncertainty propagation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Workflow:
  1. python azure_mc.py populate  input.azr
  2. (edit mc_params.yaml — adjust ranges, distributions)
  3. python azure_mc.py run       input.azr  mc_setup.yaml
  4. python azure_mc.py summary   mc_results.npz
  
  # Generate .dat files for different quantiles (without re-running MC):
  python azure_mc.py quantiles mc_results.npz -q 0.025 0.5 0.975
  # or from config:
  python azure_mc.py quantiles mc_results.npz -c mc_setup.yaml
""",
    )
    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("populate",
                        help="Dry-run: discover free parameters → YAML")
    sp.add_argument("azr_file")
    sp.add_argument("setup_out", nargs="?", default="mc_setup.yaml",
                    help="Output setup file (default: mc_setup.yaml)")
    sp.add_argument("params_out", nargs="?", default="mc_params.yaml",
                    help="Output parameters file (default: mc_params.yaml)")

    sp = sub.add_parser("run", help="Run Monte Carlo")
    sp.add_argument("azr_file")
    sp.add_argument("setup", help="Setup YAML (references params file)")
    sp.add_argument("--tmp-dir", default=None)

    sp = sub.add_parser("summary", help="Print result statistics")
    sp.add_argument("npz_file")

    sp = sub.add_parser("quantiles",
                        help="Generate .dat files with different quantiles")
    sp.add_argument("npz_file", help="Existing MC results file")
    sp.add_argument("-q", "--quantiles", type=float, nargs="+", default=None,
                    help="Quantile levels (e.g., -q 0.025 0.5 0.975)")
    sp.add_argument("-c", "--config", default=None,
                    help="YAML config with 'quantiles' (and optional 'output_prefix') - alternative to -q")
    sp.add_argument("-p", "--prefix", default=None,
                    help="Output file prefix for .dat files (default: use npz file stem)")

    args = parser.parse_args()

    if args.command == "populate":
        cmd_populate(args.azr_file, args.setup_out, args.params_out)
    elif args.command == "run":
        cmd_run(args.azr_file, args.setup, args.tmp_dir)
    elif args.command == "summary":
        cmd_summary(args.npz_file)
    elif args.command == "quantiles":
        cmd_recompute_quantiles(args.npz_file, args.quantiles, args.config, args.prefix)
    else:
        parser.print_help()

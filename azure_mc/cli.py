"""
CLI entry point — argparse setup and dispatch.
"""

from __future__ import annotations

import argparse
import sys

from .commands import (
    cmd_mc_populate, cmd_mcmc_populate,
    cmd_run, cmd_summary, cmd_recompute_quantiles,
    cmd_mcmc, cmd_mcmc_extrapolate,
)

_PROG = "azure-mc"


def main():
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="AZURE2 Monte Carlo uncertainty propagation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Workflow — random MC (extrapolation without data):
  1. {_PROG} mc populate      -i input.azr
  2. (edit params.yaml — adjust ranges, distributions)
  3. {_PROG} mc extrapolate   -i input.azr  -c mc_setup.yaml
  4. {_PROG} mc summary       -r mc_extrapolate.npz

Workflow — MCMC (fit to data, then extrapolate):
  1. {_PROG} mcmc populate      -i input.azr
  2. (edit params.yaml — adjust priors)
  3. {_PROG} mcmc fit           -i input.azr  -c mcmc_setup.yaml
  4. {_PROG} mcmc extrapolate   -i input.azr  -c mcmc_setup.yaml  --chain mcmc_chain.npz
  5. {_PROG} mcmc summary       -r mcmc_extrapolate.npz
""",
    )
    sub = parser.add_subparsers(dest="command")

    # ── mc ───────────────────────────────────────────────────────────
    mc = sub.add_parser("mc", help="Random Monte Carlo workflow")
    mc_sub = mc.add_subparsers(dest="mc_command")

    mc_pop = mc_sub.add_parser(
        "populate",
        help="Discover free parameters → mc_setup.yaml + params.yaml",
    )
    mc_pop.add_argument("-i", "--input", required=True, dest="azr_file",
                        help="AZURE2 .azr input file")
    mc_pop.add_argument("--setup-out", default="mc_setup.yaml",
                        help="Output setup file (default: mc_setup.yaml)")
    mc_pop.add_argument("--params-out", default="params.yaml",
                        help="Output parameters file (default: params.yaml)")

    mc_ext = mc_sub.add_parser(
        "extrapolate",
        help="Sample parameters and run AZURE2 extrapolation",
    )
    mc_ext.add_argument("-i", "--input", required=True, dest="azr_file",
                        help="AZURE2 .azr input file")
    mc_ext.add_argument("-c", "--config", required=True, dest="setup",
                        help="MC setup YAML")
    mc_ext.add_argument("--tmp-dir", default=None)

    mc_sum = mc_sub.add_parser(
        "summary",
        help="Print statistics and optionally write quantile .dat files",
    )
    mc_sum.add_argument("-r", "--results", required=True, dest="npz_file",
                        help="MC results .npz file")
    mc_sum.add_argument("-q", "--quantiles", type=float, nargs="+",
                        default=None,
                        help="Write .dat files for these quantile levels "
                             "(e.g. -q 0.025 0.5 0.975)")
    mc_sum.add_argument("-p", "--prefix", default=None,
                        help="Output file prefix for .dat files")

    # ── mcmc ─────────────────────────────────────────────────────────
    mcmc = sub.add_parser("mcmc", help="MCMC workflow (fit + extrapolate)")
    mcmc_sub = mcmc.add_subparsers(dest="mcmc_command")

    mcmc_pop = mcmc_sub.add_parser(
        "populate",
        help="Discover free parameters → mcmc_setup.yaml + params.yaml",
    )
    mcmc_pop.add_argument("-i", "--input", required=True, dest="azr_file",
                          help="AZURE2 .azr input file")
    mcmc_pop.add_argument("--setup-out", default="mcmc_setup.yaml",
                          help="Output setup file (default: mcmc_setup.yaml)")
    mcmc_pop.add_argument("--params-out", default="params.yaml",
                          help="Output parameters file (default: params.yaml)")

    mcmc_fit = mcmc_sub.add_parser(
        "fit",
        help="Fit parameters to data via MCMC (emcee)",
    )
    mcmc_fit.add_argument("-i", "--input", required=True, dest="azr_file",
                          help="AZURE2 .azr input file")
    mcmc_fit.add_argument("-c", "--config", required=True, dest="setup",
                          help="MCMC setup YAML")
    mcmc_fit.add_argument("--tmp-dir", default=None)

    mcmc_ext = mcmc_sub.add_parser(
        "extrapolate",
        help="Extrapolate using posterior samples from a fit",
    )
    mcmc_ext.add_argument("-i", "--input", required=True, dest="azr_file",
                          help="AZURE2 .azr input file")
    mcmc_ext.add_argument("-c", "--config", required=True, dest="setup",
                          help="MCMC setup YAML")
    mcmc_ext.add_argument("--chain", required=True, dest="mcmc_npz",
                          help="MCMC results .npz file (from 'mcmc fit')")
    mcmc_ext.add_argument("-n", "--n-draws", type=int, default=None,
                          help="Number of posterior samples "
                               "(default: min(100, chain length))")
    mcmc_ext.add_argument("--tmp-dir", default=None)

    mcmc_sum = mcmc_sub.add_parser(
        "summary",
        help="Print statistics and optionally write quantile .dat files",
    )
    mcmc_sum.add_argument("-r", "--results", required=True, dest="npz_file",
                          help="MCMC results or extrapolation .npz file")
    mcmc_sum.add_argument("-q", "--quantiles", type=float, nargs="+",
                          default=None,
                          help="Write .dat files for these quantile levels")
    mcmc_sum.add_argument("-p", "--prefix", default=None,
                          help="Output file prefix for .dat files")

    # ── dispatch ─────────────────────────────────────────────────────
    args = parser.parse_args()

    if args.command == "mc":
        if args.mc_command == "populate":
            cmd_mc_populate(args.azr_file, args.setup_out, args.params_out)
        elif args.mc_command == "extrapolate":
            cmd_run(args.azr_file, args.setup, args.tmp_dir)
        elif args.mc_command == "summary":
            _handle_summary(args)
        else:
            mc.print_help()
            sys.exit(1)

    elif args.command == "mcmc":
        if args.mcmc_command == "populate":
            cmd_mcmc_populate(args.azr_file, args.setup_out, args.params_out)
        elif args.mcmc_command == "fit":
            cmd_mcmc(args.azr_file, args.setup, args.tmp_dir)
        elif args.mcmc_command == "extrapolate":
            _validate_npz(args.mcmc_npz, {"flat_chain", "param_keys"},
                          label="MCMC fit")
            cmd_mcmc_extrapolate(args.azr_file, args.mcmc_npz, args.setup,
                                 args.n_draws, args.tmp_dir)
        elif args.mcmc_command == "summary":
            _handle_summary(args)
        else:
            mcmc.print_help()
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(0 if args.command is None else 1)


# ── helpers ──────────────────────────────────────────────────────────

def _handle_summary(args):
    """Print summary; if ``-q`` was given, also write quantile .dat files."""
    cmd_summary(args.npz_file)
    if args.quantiles:
        cmd_recompute_quantiles(args.npz_file, args.quantiles,
                                config_file=None, output_prefix=args.prefix)


def _validate_npz(path: str, required_keys: set[str], *, label: str):
    """Abort if *path* is not a valid .npz with the expected keys."""
    import numpy as np

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        print(f"Error: cannot load '{path}': {exc}", file=sys.stderr)
        sys.exit(1)

    missing = required_keys - set(data.files)
    if missing:
        print(
            f"Error: '{path}' does not look like a {label} results file "
            f"(missing keys: {', '.join(sorted(missing))}).",
            file=sys.stderr,
        )
        sys.exit(1)

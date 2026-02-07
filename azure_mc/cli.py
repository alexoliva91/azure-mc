"""
CLI entry point — argparse setup and dispatch.
"""

from __future__ import annotations

import argparse

from .commands import (
    cmd_populate, cmd_run, cmd_summary, cmd_recompute_quantiles,
    cmd_mcmc, cmd_mcmc_predict,
)


def main():
    parser = argparse.ArgumentParser(
        description="AZURE2 Monte Carlo uncertainty propagation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Workflow — random MC (extrapolation without data):
  1. python azure_mc.py populate  input.azr
  2. (edit mc_params.yaml — adjust ranges, distributions)
  3. python azure_mc.py run       input.azr  mc_setup.yaml
  4. python azure_mc.py summary   mc_results.npz

Workflow — MCMC (fit to data with emcee):
  1. python azure_mc.py populate  input.azr
  2. (edit mc_params.yaml — adjust priors)
  3. python azure_mc.py mcmc      input.azr  mc_setup.yaml
  4. python azure_mc.py predict   input.azr  mcmc_results.npz  mc_setup.yaml

Recompute quantiles without re-running:
  python azure_mc.py quantiles mc_results.npz -q 0.025 0.5 0.975
""",
    )
    sub = parser.add_subparsers(dest="command")

    # --- populate ---
    sp = sub.add_parser("populate",
                        help="Dry-run: discover free parameters → YAML")
    sp.add_argument("azr_file")
    sp.add_argument("setup_out", nargs="?", default="mc_setup.yaml",
                    help="Output setup file (default: mc_setup.yaml)")
    sp.add_argument("params_out", nargs="?", default="mc_params.yaml",
                    help="Output parameters file (default: mc_params.yaml)")

    # --- run (random MC extrapolation) ---
    sp = sub.add_parser("run", help="Run random-sampling Monte Carlo (extrapolation)")
    sp.add_argument("azr_file")
    sp.add_argument("setup", help="Setup YAML (references params file)")
    sp.add_argument("--tmp-dir", default=None)

    # --- summary ---
    sp = sub.add_parser("summary", help="Print result statistics")
    sp.add_argument("npz_file")

    # --- quantiles ---
    sp = sub.add_parser("quantiles",
                        help="Generate .dat files with different quantiles")
    sp.add_argument("npz_file", help="Existing MC results file")
    sp.add_argument("-q", "--quantiles", type=float, nargs="+", default=None,
                    help="Quantile levels (e.g., -q 0.025 0.5 0.975)")
    sp.add_argument("-c", "--config", default=None,
                    help="YAML config with 'quantiles' — alternative to -q")
    sp.add_argument("-p", "--prefix", default=None,
                    help="Output file prefix for .dat files")

    # --- mcmc (fit to data) ---
    sp = sub.add_parser("mcmc",
                        help="Run MCMC with emcee (fit to data)")
    sp.add_argument("azr_file")
    sp.add_argument("setup", help="Setup YAML (with mcmc section)")
    sp.add_argument("--tmp-dir", default=None)

    # --- predict (posterior extrapolation) ---
    sp = sub.add_parser("predict",
                        help="Extrapolate using posterior samples from MCMC")
    sp.add_argument("azr_file")
    sp.add_argument("mcmc_npz", help="MCMC results .npz file")
    sp.add_argument("setup", help="Setup YAML")
    sp.add_argument("-n", "--n-draws", type=int, default=None,
                    help="Number of posterior samples to draw (default: min(100, chain))")
    sp.add_argument("--tmp-dir", default=None)

    args = parser.parse_args()

    if args.command == "populate":
        cmd_populate(args.azr_file, args.setup_out, args.params_out)
    elif args.command == "run":
        cmd_run(args.azr_file, args.setup, args.tmp_dir)
    elif args.command == "summary":
        cmd_summary(args.npz_file)
    elif args.command == "quantiles":
        cmd_recompute_quantiles(args.npz_file, args.quantiles,
                                args.config, args.prefix)
    elif args.command == "mcmc":
        cmd_mcmc(args.azr_file, args.setup, args.tmp_dir)
    elif args.command == "predict":
        cmd_mcmc_predict(args.azr_file, args.mcmc_npz, args.setup,
                         args.n_draws, args.tmp_dir)
    else:
        parser.print_help()

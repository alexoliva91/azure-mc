"""
High-level sub-command implementations: populate, fit, extrapolate,
summary, quantiles.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from .io import (read_input_file, read_levels, get_extrap_output_files,
                 get_data_output_files, resolve_data_paths)
from .parameters import (
    discover_free_parameters, get_input_values, sample_theta,
    log_prior, initialize_walkers,
)
from .runner import run_single, log_probability

log = logging.getLogger(__name__)


def _resolve_auto_parallelism(max_workers, azure_threads, cpu_count):
    """
    Resolve 'auto' values for max_workers and azure_threads.
    
    Strategy:
    - azure_threads = 4 (good balance for linear algebra operations)
    - max_workers = cpu_count // azure_threads
    - Ensures total_threads = max_workers * azure_threads ≤ cpu_count
    
    Returns:
        tuple: (resolved_max_workers, resolved_azure_threads)
    """
    # Handle 'auto' for azure_threads
    if azure_threads == 'auto':
        # Optimal: 4 threads for BLAS/LAPACK, but adjust for small CPUs
        if cpu_count >= 8:
            azure_threads = 4
        elif cpu_count >= 4:
            azure_threads = 2
        else:
            azure_threads = 1
    
    # Handle 'auto' for max_workers
    if max_workers == 'auto':
        max_workers = max(1, cpu_count // azure_threads)
    
    return max_workers, azure_threads


def _write_params_file(
    azr_filepath: str,
    params_out: str,
    params,
    norms,
    values: list[float],
):
    """Write the shared parameters YAML file (used by both MC and MCMC)."""
    n_level_params = len(params)
    total = n_level_params + len(norms)

    param_entries = {}
    for i, p in enumerate(params):
        nom = values[i]
        frac = 0.2
        if nom != 0:
            half = abs(nom) * frac
            lo, hi = nom - half, nom + half
        else:
            lo, hi = -1.0, 1.0
        param_entries[p.key()] = {
            "description": p.description(),
            "nominal": float(nom),
            "low": float(lo),
            "high": float(hi),
            "distribution": "uniform",
        }

    for j, nf in enumerate(norms):
        idx = n_level_params + j
        nom = values[idx]
        if nom != 0:
            lo_n, hi_n = nom * 0.8, nom * 1.2
        else:
            lo_n, hi_n = -1.0, 1.0
        param_entries[nf.key()] = {
            "description": nf.description(),
            "nominal": float(nom),
            "low": float(lo_n),
            "high": float(hi_n),
            "distribution": "uniform",
        }

    with open(params_out, "w") as fh:
        fh.write("# AZURE2 — free-parameter ranges / priors\n")
        fh.write(f"# Generated from: {azr_filepath}\n")
        fh.write(f"# {n_level_params} level params + {len(norms)} norm factors "
                 f"= {total} total\n")
        fh.write("#\n")
        fh.write("# For MC:   low/high act as hard sampling bounds (clipping).\n")
        fh.write("# For MCMC: low/high act as hard prior bounds (-inf outside).\n")
        fh.write("#\n")
        fh.write("# Built-in distributions:\n")
        fh.write("#   uniform    — flat between low and high (default)\n")
        fh.write("#   gaussian   — Normal(nominal, sigma); add 'sigma' key\n")
        fh.write("#   lognormal  — LogNormal(mu, sigma); add 'mu' and 'sigma' keys\n")
        fh.write("#                defaults: mu = ln(|nominal|), sigma = 1\n")
        fh.write("#\n")
        fh.write("# Any scipy.stats distribution is also supported.  Use the\n")
        fh.write("# scipy name as 'distribution' and pass shape/loc/scale\n")
        fh.write("# parameters under the 'dist_params' key.  Examples:\n")
        fh.write("#\n")
        fh.write("#   distribution: truncnorm\n")
        fh.write("#   dist_params: {a: -2, b: 2, loc: 1.0, scale: 0.5}\n")
        fh.write("#\n")
        fh.write("#   distribution: gamma\n")
        fh.write("#   dist_params: {a: 2.0, scale: 50.0}\n")
        fh.write("#\n")
        fh.write("# 'defaults' apply to any parameter missing low/high/distribution.\n\n")
        params_file_data = {
            "defaults": {
                "fraction": 0.2,
                "distribution": "uniform",
            },
            "parameters": param_entries,
        }
        yaml.dump(params_file_data, fh, default_flow_style=False, sort_keys=False)


def _print_discovered_params(params, norms, values):
    """Print the discovered free parameters to stdout."""
    n_level_params = len(params)
    total = n_level_params + len(norms)
    print(f"  {n_level_params} level parameters + {len(norms)} norm factors "
          f"= {total} free parameters")
    print()
    print("Free parameters discovered:")
    for i, p in enumerate(params):
        print(f"  [{i:3d}] {p.key():50s}  nominal = {values[i]:15.6g}  "
              f"| {p.description()}")
    for j, nf in enumerate(norms):
        idx = n_level_params + j
        print(f"  [{idx:3d}] {nf.key():50s}  nominal = {values[idx]:15.6g}  "
              f"| {nf.description()}")


def cmd_mc_populate(azr_filepath: str, setup_out: str, params_out: str):
    """
    Populate MC config files from an .azr input.

    * *setup_out*  — MC run settings (n_samples, quantiles, …)
    * *params_out* — per-parameter ranges / distributions (shared format)
    """
    contents = read_input_file(azr_filepath)
    params, norms, addresses = discover_free_parameters(contents)
    values = get_input_values(contents, params, norms, addresses)

    # ---- shared parameters file ----
    _write_params_file(azr_filepath, params_out, params, norms, values)

    # ---- setup file (MC-only) ----
    setup_cfg = {
        "azure2_exe": "AZURE2",
        "use_brune": True,
        "use_gsl": True,
        "n_samples": 100,
        "max_workers": "auto",
        "azure_threads": "auto",
        "seed": 42,
        "keep_tmp": False,
        "timeout": 0,
        "output_file": "mc_extrapolate.npz",
        "params_file": params_out,
        "quantiles": [0.16, 0.50, 0.84],
        # "output_prefix": "mc_results",  # Optional: prefix for .dat files
    }

    with open(setup_out, "w") as fh:
        fh.write("# AZURE2 Monte Carlo — MC run setup\n")
        fh.write("# -----------------------------------\n")
        fh.write(f"# Generated from: {azr_filepath}\n")
        fh.write("#\n")
        fh.write("# Workflow:\n")
        fh.write(f"#   1. python -m azure_mc mc populate     -i {azr_filepath}\n")
        fh.write(f"#   2. (edit {params_out} — adjust sampling ranges)\n")
        fh.write(f"#   3. python -m azure_mc mc extrapolate  -i {azr_filepath} -c {setup_out}\n")
        fh.write(f"#   4. python -m azure_mc mc summary      -r mc_extrapolate.npz -q 0.16 0.50 0.84\n\n")
        yaml.dump(setup_cfg, fh, default_flow_style=False, sort_keys=False)

    print(f"MC setup  written to {setup_out}")
    print(f"Params    written to {params_out}")
    _print_discovered_params(params, norms, values)


def cmd_mcmc_populate(azr_filepath: str, setup_out: str, params_out: str):
    """
    Populate MCMC config files from an .azr input.

    * *setup_out*  — MCMC settings (n_walkers, n_steps, …)
    * *params_out* — per-parameter ranges / priors (shared format)
    """
    contents = read_input_file(azr_filepath)
    params, norms, addresses = discover_free_parameters(contents)
    values = get_input_values(contents, params, norms, addresses)

    n_level_params = len(params)
    total = n_level_params + len(norms)

    # ---- shared parameters file ----
    _write_params_file(azr_filepath, params_out, params, norms, values)

    # ---- setup file (MCMC-only, flat structure) ----
    setup_cfg = {
        "azure2_exe": "AZURE2",
        "use_brune": True,
        "use_gsl": True,
        "max_workers": "auto",
        "azure_threads": "auto",
        "seed": 42,
        "timeout": 0,
        "params_file": params_out,
        "quantiles": [0.16, 0.50, 0.84],
        "n_walkers": max(2 * total + 2, 32),
        "n_steps": 1000,
        "n_burn": 200,
        "thin": 1,
        "init_spread": 1e-4,
        "output_file": "mcmc_chain.npz",
        "extrapolate_output_file": "mcmc_extrapolate.npz",
        "progress": True,
    }

    with open(setup_out, "w") as fh:
        fh.write("# AZURE2 MCMC — run setup\n")
        fh.write("# ------------------------\n")
        fh.write(f"# Generated from: {azr_filepath}\n")
        fh.write("#\n")
        fh.write("# Workflow:\n")
        fh.write(f"#   1. python -m azure_mc mcmc populate     -i {azr_filepath}\n")
        fh.write(f"#   2. (edit {params_out} — adjust priors)\n")
        fh.write(f"#   3. python -m azure_mc mcmc fit          -i {azr_filepath} -c {setup_out}\n")
        fh.write(f"#   4. python -m azure_mc mcmc extrapolate  -i {azr_filepath} -c {setup_out} --chain mcmc_chain.npz\n")
        fh.write(f"#   5. python -m azure_mc mcmc summary      -r mcmc_extrapolate.npz\n\n")
        yaml.dump(setup_cfg, fh, default_flow_style=False, sort_keys=False)

    print(f"MCMC setup written to {setup_out}")
    print(f"Params     written to {params_out}")
    _print_discovered_params(params, norms, values)


def cmd_run(
    azr_filepath: str,
    setup_filepath: str,
    tmp_dir: str | None = None,
):
    """Run the Monte Carlo, collect distributions, compute quantiles."""
    with open(setup_filepath, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    n_samples = cfg.get("n_samples", 100)
    max_workers = cfg.get("max_workers", os.cpu_count())
    azure_threads = cfg.get("azure_threads", 1)
    
    # Resolve 'auto' values
    cpu_count = os.cpu_count() or 1
    max_workers, azure_threads = _resolve_auto_parallelism(
        max_workers, azure_threads, cpu_count
    )
    
    azure2_cmd = cfg.get("azure2_exe", "AZURE2")
    if not shutil.which(azure2_cmd):
        log.error("AZURE2 executable '%s' not found in PATH. "
                  "Set 'azure2_exe' in setup YAML or install AZURE2.",
                  azure2_cmd)
        sys.exit(1)
    use_brune = cfg.get("use_brune", True)
    use_gsl = cfg.get("use_gsl", True)
    seed = cfg.get("seed", 42)
    keep_tmp = cfg.get("keep_tmp", False)
    timeout = cfg.get("timeout", 0)
    output_file = cfg.get("output_file", "mc_extrapolate.npz")
    quantiles_list = cfg.get("quantiles", [0.16, 0.50, 0.84])

    # Load parameters from separate file
    params_filepath = cfg.get("params_file", "parameters.yaml")
    # Resolve relative to setup file directory
    if not os.path.isabs(params_filepath):
        params_filepath = os.path.join(
            os.path.dirname(os.path.abspath(setup_filepath)), params_filepath
        )
    with open(params_filepath, "r") as fh:
        params_data = yaml.safe_load(fh) or {}
    user_params = params_data.get("parameters", params_data)
    defaults = params_data.get("defaults", {})
    default_frac = defaults.get("fraction", 0.2)
    default_dist = defaults.get("distribution", "uniform")

    # Parse .azr
    contents = read_input_file(azr_filepath)
    params, norms, addresses = discover_free_parameters(contents)
    nominals_list = get_input_values(contents, params, norms, addresses)
    nominals = np.array(nominals_list)
    levels = read_levels(contents)
    extrap_files = get_extrap_output_files(contents)
    n_level = len(params)

    log.info("Parsed %s", azr_filepath)
    log.info("  %d level params + %d norm factors = %d free",
             n_level, len(norms), len(nominals))
    log.info("  Extrap output files: %s", extrap_files)
    log.info("Parallelism: %d workers × %d Azure threads = %d total threads",
             max_workers, azure_threads, max_workers * azure_threads)

    # Build per-parameter range dicts
    all_keys = [p.key() for p in params] + [nf.key() for nf in norms]

    # Validate parameter keys match between .azr and YAML
    discovered_keys = set(all_keys)
    yaml_keys = set(user_params.keys())
    extra = yaml_keys - discovered_keys
    missing = discovered_keys - yaml_keys
    if extra:
        log.warning("Params in YAML not found in .azr (ignored): %s",
                    sorted(extra))
    if missing:
        log.warning("Params in .azr not found in YAML (using defaults): %s",
                    sorted(missing))

    ranges = _build_ranges(all_keys, nominals, user_params,
                           default_frac, default_dist)

    # Sampling
    rng = np.random.default_rng(seed)
    all_theta = np.empty((n_samples, len(nominals)))
    for i in range(n_samples):
        all_theta[i] = sample_theta(nominals, ranges, rng)

    # Tempdir
    if tmp_dir is None:
        import tempfile
        base_tmp = tempfile.mkdtemp(prefix="azure_mc_")
    else:
        base_tmp = tmp_dir
        os.makedirs(base_tmp, exist_ok=True)
    log.info("Temporary workspace: %s", base_tmp)

    log.info("Launching %d runs (%d workers) ...", n_samples, max_workers)
    results_dict: dict[int, dict[str, np.ndarray]] = {}
    n_failed = 0

    # Signal handling for graceful shutdown
    shutdown_requested = False
    def _handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        log.warning("Shutdown requested (signal %d), finishing current runs...", signum)

    old_sigint = signal.signal(signal.SIGINT, _handle_shutdown)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in range(n_samples):
                theta_levels = all_theta[i, :n_level]
                norm_updates = None
                if norms:
                    norm_updates = [
                        (nf.index, float(all_theta[i, n_level + j]))
                        for j, nf in enumerate(norms)
                    ]
                fut = executor.submit(
                    run_single,
                    i,
                    contents,
                    levels,
                    addresses,
                    theta_levels,
                    extrap_files,
                    azure2_cmd,
                    use_brune,
                    use_gsl,
                    base_tmp,
                    keep_tmp,
                    timeout,
                    norm_updates=norm_updates,
                    azure_threads=azure_threads,
                )
                futures[fut] = i

            for fut in as_completed(futures):
                run_id, data, msg = fut.result()
                if data is not None:
                    results_dict[run_id] = data
                else:
                    n_failed += 1
                    log.warning("  %s", msg)
                done = len(results_dict) + n_failed
                if done % max(1, n_samples // 20) == 0:
                    log.info("  Progress: %d/%d  (%d ok, %d failed)",
                             done, n_samples, len(results_dict), n_failed)
                if shutdown_requested:
                    log.warning("Cancelling remaining %d runs...",
                                n_samples - done)
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    log.info("Done: %d/%d succeeded, %d failed",
             len(results_dict), n_samples, n_failed)

    if not results_dict:
        log.error("No successful runs — cannot produce output.")
        if not keep_tmp:
            shutil.rmtree(base_tmp, ignore_errors=True)
        return

    # ------------------------------------------------------------------
    # Assemble per-channel bucket arrays.
    # Each run returns (n_pts, 3) per channel: [energy, xs, sfactor].
    # Build two buckets per channel — one for cross section, one for
    # S-factor — then compute quantiles and write output files.
    # ------------------------------------------------------------------
    sorted_ids = sorted(results_dict.keys())
    n_ok = len(sorted_ids)

    channel_names = sorted(
        {ch for per_file in results_dict.values() for ch in per_file}
    )
    log.info("Channels found: %s", channel_names)

    save_dict: dict[str, np.ndarray] = {  # type: ignore[arg-type]  # keys won't clash with savez params
        "samples": all_theta[sorted_ids],
        "param_keys": np.array(all_keys),
        "param_nominals": nominals,
        "quantile_levels": np.array(quantiles_list),
        "channel_names": np.array(channel_names),
    }

    out_dir = Path(output_file).resolve().parent
    out_stem = Path(output_file).stem

    for ch_name in channel_names:
        # Reference energy grid from first successful run
        ref_run = None
        for rid in sorted_ids:
            if ch_name in results_dict[rid]:
                ref_run = results_dict[rid][ch_name]
                break
        if ref_run is None:
            continue

        energies = ref_run[:, 0]
        n_pts = len(energies)

        # Build buckets: (n_ok, n_pts) for xs and sfactor
        bucket_xs = np.full((n_ok, n_pts), np.nan)
        bucket_sf = np.full((n_ok, n_pts), np.nan)
        for idx, rid in enumerate(sorted_ids):
            if ch_name not in results_dict[rid]:
                continue
            arr = results_dict[rid][ch_name]  # (n_pts, 3)
            if arr.shape[0] == n_pts:
                bucket_xs[idx] = arr[:, 1]  # cross section
                bucket_sf[idx] = arr[:, 2]  # s-factor
            else:
                log.warning("Run %d, channel %s: %d pts (expected %d), skipping",
                            rid, ch_name, arr.shape[0], n_pts)

        # Keep only rows with data
        valid_mask = ~np.all(np.isnan(bucket_xs), axis=1)
        bucket_xs_v = bucket_xs[valid_mask]
        bucket_sf_v = bucket_sf[valid_mask]
        n_valid = bucket_xs_v.shape[0]
        log.info("  Channel %-45s  %d energy pts x %d valid runs",
                 ch_name, n_pts, n_valid)

        if n_valid == 0:
            continue

        # Quantiles for both quantities: shape (n_quantiles, n_pts)
        q_xs = np.nanquantile(bucket_xs_v, quantiles_list, axis=0)
        q_sf = np.nanquantile(bucket_sf_v, quantiles_list, axis=0)

        # Store in npz
        safe_ch = ch_name.replace(".", "_").replace("=", "_")
        save_dict[f"{safe_ch}/energies"] = energies
        save_dict[f"{safe_ch}/bucket_xs"] = bucket_xs_v
        save_dict[f"{safe_ch}/bucket_sf"] = bucket_sf_v

        # Write one output file per quantile level.
        # Each file has 3 columns: Energy(MeV)  CrossSection(b)  S-factor(MeV*b)
        for qi, q in enumerate(quantiles_list):
            q_tag = f"Q{q * 100:g}"
            dat_name = f"{out_stem}_{safe_ch}_{q_tag}.dat"
            dat_path = out_dir / dat_name
            with open(dat_path, "w") as fh:
                fh.write(f"# Channel  : {ch_name}\n")
                fh.write(f"# Quantile : {q_tag}% ({n_valid} valid runs)\n")
                fh.write(f"# Energy(MeV)  CrossSection(b)  S-factor(MeV*b)\n")
                for j in range(n_pts):
                    fh.write(f"{energies[j]:.6e}  {q_xs[qi, j]:.6e}  {q_sf[qi, j]:.6e}\n")
            log.info("    -> %s", dat_path)

    np.savez(file=output_file, **save_dict)  # type: ignore[arg-type]
    log.info("Saved %s  (%d channels, %d runs)", output_file,
             len(channel_names), n_ok)

    if not keep_tmp:
        shutil.rmtree(base_tmp, ignore_errors=True)


def cmd_recompute_quantiles(
    npz_file: str,
    quantiles: list[float] | None = None,
    config_file: str | None = None,
    output_prefix: str | None = None,
):
    """Generate .dat files with different quantiles from existing MC results."""
    # Load from config file if provided
    if config_file:
        with open(config_file, "r") as fh:
            cfg = yaml.safe_load(fh) or {}

        # Get quantiles from config if not provided via -q
        if quantiles is None:
            quantiles = cfg.get("quantiles")
            if quantiles is None:
                log.error("Config file '%s' does not contain 'quantiles' key", config_file)
                sys.exit(1)
            log.info("Using quantiles from config: %s", config_file)
        else:
            log.info("Using quantiles from -q (ignoring config file)")

        # Get output prefix from config if not provided via -p
        if output_prefix is None:
            output_prefix = cfg.get("output_prefix")

    if quantiles is None:
        log.error("No quantiles provided. Use -q or -c/--config")
        sys.exit(1)

    data = np.load(npz_file, allow_pickle=True)
    channel_names = data["channel_names"]
    n_runs = data["samples"].shape[0]

    out_dir = Path(npz_file).resolve().parent
    if output_prefix is None:
        output_prefix = Path(npz_file).stem

    log.info("Extracting quantiles from %s", npz_file)
    log.info("  Quantiles: %s", quantiles)
    log.info("  MC runs: %d", n_runs)
    log.info("  Output prefix: %s", output_prefix)

    for ch_name in channel_names:
        safe_ch = str(ch_name).replace(".", "_").replace("=", "_")
        key_e = f"{safe_ch}/energies"
        key_bx = f"{safe_ch}/bucket_xs"
        key_bs = f"{safe_ch}/bucket_sf"

        if key_bx not in data:
            continue

        energies = data[key_e]
        bucket_xs = data[key_bx]
        bucket_sf = data[key_bs]
        n_valid = bucket_xs.shape[0]
        n_pts = len(energies)

        # Compute requested quantiles
        q_xs = np.nanquantile(bucket_xs, quantiles, axis=0)
        q_sf = np.nanquantile(bucket_sf, quantiles, axis=0)

        log.info("  Channel %-45s  %d pts x %d runs",
                 ch_name, n_pts, n_valid)

        # Write output files for each quantile
        for qi, q in enumerate(quantiles):
            q_tag = f"Q{q * 100:g}"
            dat_name = f"{output_prefix}_{safe_ch}_{q_tag}.dat"
            dat_path = out_dir / dat_name
            with open(dat_path, "w") as fh:
                fh.write(f"# Channel  : {ch_name}\n")
                fh.write(f"# Quantile : {q_tag}% ({n_valid} valid runs)\n")
                fh.write(f"# Energy(MeV)  CrossSection(b)  S-factor(MeV*b)\n")
                for j in range(n_pts):
                    fh.write(f"{energies[j]:.6e}  {q_xs[qi, j]:.6e}  {q_sf[qi, j]:.6e}\n")
            log.info("    -> %s", dat_path)

    log.info("Done. Generated .dat files for %d quantiles across %d channels",
             len(quantiles), len(channel_names))


def cmd_summary(npz_file: str):
    """Print summary statistics from an MC results file."""
    data = np.load(npz_file, allow_pickle=True)
    keys = data["param_keys"]
    nominals = data["param_nominals"]
    q_levels = data["quantile_levels"]
    channel_names = data["channel_names"]
    n_runs = data["samples"].shape[0]

    print(f"MC runs    : {n_runs}")
    print(f"Parameters : {len(keys)}")
    print(f"Quantiles  : {list(q_levels)}")
    print(f"Channels   : {len(channel_names)}")
    print()

    for ch_name in channel_names:
        safe_ch = str(ch_name).replace(".", "_").replace("=", "_")
        key_e = f"{safe_ch}/energies"
        key_bx = f"{safe_ch}/bucket_xs"
        key_bs = f"{safe_ch}/bucket_sf"
        if key_e not in data:
            continue

        energies = data[key_e]
        bucket_xs = data[key_bx]
        bucket_sf = data[key_bs]

        # Compute quantiles on the fly
        q_xs = np.nanquantile(bucket_xs, q_levels, axis=0)
        q_sf = np.nanquantile(bucket_sf, q_levels, axis=0)

        print(f"--- Channel: {ch_name} ---")
        print(f"  Energy pts : {len(energies)}, "
              f"[{energies.min():.6f}, {energies.max():.6f}] MeV")
        print(f"  Valid runs : {bucket_xs.shape[0]}")
        print()

        # Table: for each quantile show a few energy-point rows
        for qi, q in enumerate(q_levels):
            q_tag = f"Q{q * 100:g}%"
            print(f"  {q_tag}:")
            print(f"    {'E (MeV)':>12s}  {'XS (b)':>14s}  {'S (MeV b)':>14s}")
            n_pts = len(energies)
            idxs = sorted(set(
                list(range(min(3, n_pts))) +
                list(range(max(0, n_pts - 3), n_pts))
            ))
            prev_i = -1
            for i in idxs:
                if prev_i >= 0 and i - prev_i > 1:
                    print(f"    {'...':>12s}")
                print(f"    {energies[i]:12.6f}  {q_xs[qi, i]:14.6e}  {q_sf[qi, i]:14.6e}")
                prev_i = i
            if n_pts > 6:
                print(f"    ({n_pts - 6} more rows)")


# ======================================================================
# MCMC commands
# ======================================================================

def _build_ranges(
    all_keys: list[str],
    nominals: np.ndarray,
    user_params: dict,
    default_frac: float,
    default_dist: str,
) -> list[dict]:
    """Build per-parameter range dicts, shared by ``cmd_run`` and ``cmd_mcmc``.

    All user-specified keys (including ``dist_params``, ``mu``, ``sigma``,
    etc.) are forwarded so that ``sample_theta`` and ``log_prior`` can
    use them.
    """
    ranges: list[dict] = []
    for i, key in enumerate(all_keys):
        nom = nominals[i]
        if key in user_params:
            r = dict(user_params[key])
            dist = r.get("distribution", default_dist)
            # Only inject default low/high for uniform (it needs them);
            # other distributions work without bounds.
            if dist == "uniform" and ("low" not in r or "high" not in r):
                half = abs(nom) * default_frac if nom != 0 else 1.0
                r.setdefault("low", nom - half)
                r.setdefault("high", nom + half)
            r.setdefault("distribution", default_dist)
            r.setdefault("nominal", nom)
        else:
            half = abs(nom) * default_frac if nom != 0 else 1.0
            r = {"low": nom - half, "high": nom + half,
                 "distribution": default_dist, "nominal": nom}
        ranges.append(r)
    return ranges


def cmd_mcmc(
    azr_filepath: str,
    setup_filepath: str,
    tmp_dir: str | None = None,
):
    """Run MCMC with *emcee* using AZURE2 χ² as the likelihood.

    The parameters in ``parameters.yaml`` are used as priors.
    AZURE2 is invoked in *Calculate With Data* mode (choice 1) to
    evaluate the likelihood at each walker position proposed by emcee.
    """
    try:
        import emcee
    except ImportError:
        log.error("emcee is required for MCMC.  Install with:  pip install emcee")
        sys.exit(1)
    from multiprocessing import Pool

    # ---- load config (flat structure — no nested 'mcmc' block) ----
    with open(setup_filepath, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    azure2_cmd = cfg.get("azure2_exe", "AZURE2")
    if not shutil.which(azure2_cmd):
        log.error("AZURE2 executable '%s' not found in PATH.", azure2_cmd)
        sys.exit(1)
    use_brune = cfg.get("use_brune", True)
    use_gsl = cfg.get("use_gsl", True)
    max_workers = cfg.get("max_workers", os.cpu_count())
    azure_threads = cfg.get("azure_threads", 1)
    
    # Resolve 'auto' values
    cpu_count = os.cpu_count() or 1
    max_workers, azure_threads = _resolve_auto_parallelism(
        max_workers, azure_threads, cpu_count
    )
    
    seed = cfg.get("seed", 42)
    timeout = cfg.get("timeout", 0)
    quantiles_list = cfg.get("quantiles", [0.16, 0.50, 0.84])

    n_walkers = cfg.get("n_walkers", 32)
    n_steps = cfg.get("n_steps", 1000)
    n_burn = cfg.get("n_burn", 200)
    thin = cfg.get("thin", 1)
    init_spread = cfg.get("init_spread", 1e-4)
    output_file = cfg.get("output_file", "mcmc_chain.npz")
    progress = cfg.get("progress", True)

    # ---- load parameter ranges ----
    params_filepath = cfg.get("params_file", "parameters.yaml")
    if not os.path.isabs(params_filepath):
        params_filepath = os.path.join(
            os.path.dirname(os.path.abspath(setup_filepath)), params_filepath
        )
    with open(params_filepath, "r") as fh:
        params_data = yaml.safe_load(fh) or {}
    user_params = params_data.get("parameters", params_data)
    defaults = params_data.get("defaults", {})
    default_frac = defaults.get("fraction", 0.2)
    default_dist = defaults.get("distribution", "uniform")

    # ---- parse .azr & resolve data paths ----
    contents = read_input_file(azr_filepath)
    azr_base_dir = os.path.dirname(os.path.abspath(azr_filepath))
    contents = resolve_data_paths(contents, azr_base_dir)

    params, norms, addresses = discover_free_parameters(contents)
    nominals_list = get_input_values(contents, params, norms, addresses)
    nominals = np.array(nominals_list)
    levels = read_levels(contents)
    n_level = len(params)
    ndim = len(nominals)
    data_output_files = get_data_output_files(contents)

    all_keys = [p.key() for p in params] + [nf.key() for nf in norms]
    ranges = _build_ranges(all_keys, nominals, user_params,
                           default_frac, default_dist)

    # ---- sanity checks ----
    if n_walkers < 2 * ndim:
        old_nw = n_walkers
        n_walkers = 2 * ndim + 2
        log.warning("n_walkers (%d) < 2*ndim (%d).  Increased to %d.",
                    old_nw, 2 * ndim, n_walkers)
    if not data_output_files:
        log.error("No <segmentsData> found in %s — "
                  "no data to fit against.", azr_filepath)
        sys.exit(1)

    log.info("MCMC setup  : %d walkers, %d steps, %d params (n_burn=%d, thin=%d)",
             n_walkers, n_steps, ndim, n_burn, thin)
    log.info("  Level params: %d,  Norm factors: %d", n_level, len(norms))
    log.info("  Data output files: %s", data_output_files)
    log.info("Parallelism: %d workers × %d Azure threads = %d total threads",
             max_workers, azure_threads, max_workers * azure_threads)

    # ---- initialise walkers ----
    rng = np.random.default_rng(seed)
    p0 = initialize_walkers(nominals, ranges, n_walkers, rng,
                            spread=init_spread)

    # ---- temp workspace ----
    if tmp_dir is None:
        import tempfile
        base_tmp = tempfile.mkdtemp(prefix="azure_mcmc_")
    else:
        base_tmp = tmp_dir
        os.makedirs(base_tmp, exist_ok=True)
    log.info("Temporary workspace: %s", base_tmp)

    # ---- run sampler ----
    lp_args = (
        contents, levels, addresses, ranges,
        n_level, norms, data_output_files,
        azure2_cmd, use_brune, use_gsl, base_tmp, timeout,
        azure_threads,
    )

    # Graceful shutdown handling
    shutdown_requested = False

    def _handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        log.warning("Shutdown requested (signal %d) — "
                    "finishing current step...", signum)

    old_sigint = signal.signal(signal.SIGINT, _handle_shutdown)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_shutdown)

    pool = None
    try:
        if max_workers > 1:
            pool = Pool(max_workers)
            sampler = emcee.EnsembleSampler(
                n_walkers, ndim, log_probability,
                args=lp_args, pool=pool,
            )
        else:
            sampler = emcee.EnsembleSampler(
                n_walkers, ndim, log_probability,
                args=lp_args,
            )

        log.info("Starting MCMC ...")
        for step_result in sampler.sample(p0, iterations=n_steps,
                                          progress=progress):
            if shutdown_requested:
                log.warning("Stopping MCMC early at step %d.", sampler.iteration)
                break
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    n_completed = sampler.iteration
    log.info("MCMC completed %d/%d steps.  Mean acceptance: %.3f",
             n_completed, n_steps,
             np.mean(sampler.acceptance_fraction))

    # ---- autocorrelation ----
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        log.info("Autocorrelation times (mean %.1f): %s",
                 np.mean(tau), np.round(tau, 1))
    except emcee.autocorr.AutocorrError:
        tau = np.full(ndim, np.nan)
        log.warning("Could not estimate autocorrelation (chain too short)")

    # ---- save ----
    chain = sampler.get_chain()                    # (n_steps, n_walkers, ndim)
    log_prob = sampler.get_log_prob()              # (n_steps, n_walkers)

    actual_burn = min(n_burn, n_completed - 1)
    flat_chain: np.ndarray = np.asarray(
        sampler.get_chain(discard=actual_burn, thin=thin, flat=True)
    )
    flat_log_prob = sampler.get_log_prob(discard=actual_burn, thin=thin,
                                        flat=True)

    posterior_q = np.quantile(flat_chain, quantiles_list, axis=0)  # (n_q, ndim)

    save_dict = {
        "chain": chain,
        "log_prob": log_prob,
        "flat_chain": flat_chain,
        "flat_log_prob": flat_log_prob,
        "param_keys": np.array(all_keys),
        "param_nominals": nominals,
        "acceptance_fraction": sampler.acceptance_fraction,
        "autocorr_time": tau,
        "n_burn": np.array(actual_burn),
        "thin": np.array(thin),
        "quantile_levels": np.array(quantiles_list),
        "posterior_quantiles": posterior_q,
    }

    np.savez(file=output_file, **save_dict)
    log.info("Saved MCMC results → %s  (%d effective samples)",
             output_file, flat_chain.shape[0])

    # ---- posterior summary ----
    print(f"\nMCMC Posterior Summary "
          f"(steps={n_completed}, burn-in={actual_burn}, thin={thin}, "
          f"effective={flat_chain.shape[0]}):")
    print(f"  Mean acceptance fraction: "
          f"{np.mean(sampler.acceptance_fraction):.3f}")
    print()
    header = f"{'Parameter':<50s}"
    for q in quantiles_list:
        header += f"  {'Q'+str(q*100)+'%':>12s}"
    print(header)
    print("-" * (50 + 14 * len(quantiles_list)))
    for i, key in enumerate(all_keys):
        row = f"{key:<50s}"
        for qi in range(len(quantiles_list)):
            row += f"  {posterior_q[qi, i]:>12.6g}"
        print(row)

    # ---- cleanup ----
    shutil.rmtree(base_tmp, ignore_errors=True)


def cmd_mcmc_extrapolate(
    azr_filepath: str,
    mcmc_npz_file: str,
    setup_filepath: str,
    n_draws: int | None = None,
    tmp_dir: str | None = None,
):
    """Draw posterior samples from an MCMC chain and run AZURE2 extrapolation.

    This produces the same kind of quantile ``.dat`` files as ``cmd_run``
    but using parameter vectors drawn from the posterior distribution.
    """
    # ---- load config (flat structure) ----
    with open(setup_filepath, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    azure2_cmd = cfg.get("azure2_exe", "AZURE2")
    if not shutil.which(azure2_cmd):
        log.error("AZURE2 executable '%s' not found in PATH.", azure2_cmd)
        sys.exit(1)
    use_brune = cfg.get("use_brune", True)
    use_gsl = cfg.get("use_gsl", True)
    max_workers = cfg.get("max_workers", 4)
    seed = cfg.get("seed", 42)
    keep_tmp = cfg.get("keep_tmp", False)
    timeout = cfg.get("timeout", 600)
    quantiles_list = cfg.get("quantiles", [0.16, 0.50, 0.84])
    output_file = cfg.get("extrapolate_output_file", "mcmc_extrapolate.npz")

    # ---- load MCMC chain ----
    mcmc_data = np.load(mcmc_npz_file, allow_pickle=True)
    flat_chain = mcmc_data["flat_chain"]
    all_keys_saved = list(mcmc_data["param_keys"])
    n_available = flat_chain.shape[0]

    if n_draws is None:
        n_draws = min(100, n_available)
    n_draws = min(n_draws, n_available)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_available, size=n_draws, replace=False)
    all_theta = flat_chain[indices]

    log.info("Drawing %d posterior samples from %s (%d available)",
             n_draws, mcmc_npz_file, n_available)

    # ---- parse .azr ----
    contents = read_input_file(azr_filepath)
    params, norms, addresses = discover_free_parameters(contents)
    nominals_list = get_input_values(contents, params, norms, addresses)
    nominals = np.array(nominals_list)
    levels = read_levels(contents)
    extrap_files = get_extrap_output_files(contents)
    n_level = len(params)

    all_keys = [p.key() for p in params] + [nf.key() for nf in norms]
    if all_keys != all_keys_saved:
        log.warning("Parameter keys in .azr differ from MCMC chain.  "
                    "Ensure the same .azr file was used.")

    if not extrap_files:
        log.error("No <segmentsTest> found in %s — "
                  "nothing to extrapolate.", azr_filepath)
        sys.exit(1)

    # ---- temp workspace ----
    if tmp_dir is None:
        import tempfile
        base_tmp = tempfile.mkdtemp(prefix="azure_mcmc_pred_")
    else:
        base_tmp = tmp_dir
        os.makedirs(base_tmp, exist_ok=True)
    log.info("Temporary workspace: %s", base_tmp)

    # ---- run extrapolations ----
    log.info("Launching %d extrapolation runs (%d workers) ...",
             n_draws, max_workers)

    results_dict: dict[int, dict[str, np.ndarray]] = {}
    n_failed = 0

    shutdown_requested = False

    def _handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        log.warning("Shutdown requested")

    old_sigint = signal.signal(signal.SIGINT, _handle_shutdown)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in range(n_draws):
                theta_levels = all_theta[i, :n_level]
                norm_updates = None
                if norms:
                    norm_updates = [
                        (nf.index, float(all_theta[i, n_level + j]))
                        for j, nf in enumerate(norms)
                    ]
                fut = executor.submit(
                    run_single, i,
                    contents, levels, addresses, theta_levels,
                    extrap_files, azure2_cmd, use_brune, use_gsl,
                    base_tmp, keep_tmp, timeout,
                    norm_updates=norm_updates,
                )
                futures[fut] = i

            for fut in as_completed(futures):
                run_id, data, msg = fut.result()
                if data is not None:
                    results_dict[run_id] = data
                else:
                    n_failed += 1
                    log.warning("  %s", msg)
                done = len(results_dict) + n_failed
                if done % max(1, n_draws // 20) == 0:
                    log.info("  Progress: %d/%d  (%d ok, %d failed)",
                             done, n_draws, len(results_dict), n_failed)
                if shutdown_requested:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    log.info("Done: %d/%d succeeded, %d failed",
             len(results_dict), n_draws, n_failed)

    if not results_dict:
        log.error("No successful runs.")
        if not keep_tmp:
            shutil.rmtree(base_tmp, ignore_errors=True)
        return

    # ---- assemble results (same logic as cmd_run) ----
    sorted_ids = sorted(results_dict.keys())
    n_ok = len(sorted_ids)

    channel_names = sorted(
        {ch for per_file in results_dict.values() for ch in per_file}
    )
    log.info("Channels found: %s", channel_names)

    save_dict: dict[str, np.ndarray] = {
        "samples": all_theta[sorted_ids] if max(sorted_ids) < len(all_theta) else all_theta[:n_ok],
        "param_keys": np.array(all_keys),
        "param_nominals": nominals,
        "quantile_levels": np.array(quantiles_list),
        "channel_names": np.array(channel_names),
        "source": np.array("mcmc_extrapolate"),
    }

    out_dir = Path(output_file).resolve().parent
    out_stem = Path(output_file).stem

    for ch_name in channel_names:
        ref_run = None
        for rid in sorted_ids:
            if ch_name in results_dict[rid]:
                ref_run = results_dict[rid][ch_name]
                break
        if ref_run is None:
            continue

        energies = ref_run[:, 0]
        n_pts = len(energies)

        bucket_xs = np.full((n_ok, n_pts), np.nan)
        bucket_sf = np.full((n_ok, n_pts), np.nan)
        for idx, rid in enumerate(sorted_ids):
            if ch_name not in results_dict[rid]:
                continue
            arr = results_dict[rid][ch_name]
            if arr.shape[0] == n_pts:
                bucket_xs[idx] = arr[:, 1]
                bucket_sf[idx] = arr[:, 2]

        valid_mask = ~np.all(np.isnan(bucket_xs), axis=1)
        bucket_xs_v = bucket_xs[valid_mask]
        bucket_sf_v = bucket_sf[valid_mask]
        n_valid = bucket_xs_v.shape[0]
        log.info("  Channel %-45s  %d pts x %d valid",
                 ch_name, n_pts, n_valid)

        if n_valid == 0:
            continue

        q_xs = np.nanquantile(bucket_xs_v, quantiles_list, axis=0)
        q_sf = np.nanquantile(bucket_sf_v, quantiles_list, axis=0)

        safe_ch = ch_name.replace(".", "_").replace("=", "_")
        save_dict[f"{safe_ch}/energies"] = energies
        save_dict[f"{safe_ch}/bucket_xs"] = bucket_xs_v
        save_dict[f"{safe_ch}/bucket_sf"] = bucket_sf_v

        for qi, q in enumerate(quantiles_list):
            q_tag = f"Q{q * 100:g}"
            dat_name = f"{out_stem}_{safe_ch}_{q_tag}.dat"
            dat_path = out_dir / dat_name
            with open(dat_path, "w") as fh:
                fh.write(f"# Channel  : {ch_name}\n")
                fh.write(f"# Source   : MCMC posterior extrapolation\n")
                fh.write(f"# Quantile : {q_tag}% ({n_valid} valid runs)\n")
                fh.write(f"# Energy(MeV)  CrossSection(b)  S-factor(MeV*b)\n")
                for j in range(n_pts):
                    fh.write(f"{energies[j]:.6e}  {q_xs[qi, j]:.6e}  "
                             f"{q_sf[qi, j]:.6e}\n")
            log.info("    -> %s", dat_path)

    np.savez(file=output_file, **save_dict)  # type: ignore[arg-type]
    log.info("Saved %s  (%d channels, %d runs)", output_file,
             len(channel_names), n_ok)

    if not keep_tmp:
        shutil.rmtree(base_tmp, ignore_errors=True)

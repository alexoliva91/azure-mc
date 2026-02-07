"""
High-level sub-command implementations: populate, run, summary, quantiles.
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

from .io import read_input_file, read_levels, get_extrap_output_files
from .parameters import discover_free_parameters, get_input_values, sample_theta
from .runner import run_single

log = logging.getLogger(__name__)


def cmd_populate(azr_filepath: str, setup_out: str, params_out: str):
    """
    Dry-run: parse .azr, discover free parameters, write two YAML files.

    * *setup_out*  — run settings (n_samples, quantiles, defaults, …)
    * *params_out* — per-parameter ranges / distributions
    """
    contents = read_input_file(azr_filepath)
    params, norms, addresses = discover_free_parameters(contents)
    values = get_input_values(contents, params, norms, addresses)

    n_level_params = len(params)
    total = n_level_params + len(norms)

    # ---- parameters file ----
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
        param_entries[nf.key()] = {
            "description": nf.description(),
            "nominal": float(nom),
            "low": float(nom * 0.8),
            "high": float(nom * 1.2),
            "distribution": "uniform",
        }

    with open(params_out, "w") as fh:
        fh.write("# AZURE2 MC — free-parameter ranges\n")
        fh.write(f"# Generated from: {azr_filepath}\n")
        fh.write(f"# {n_level_params} level params + {len(norms)} norm factors "
                 f"= {total} total\n")
        fh.write("#\n")
        fh.write("# 'distribution' can be 'uniform' or 'gaussian'.\n")
        fh.write("# For 'gaussian', add a 'sigma' key (std deviation).\n")
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

    # ---- setup file ----
    setup_cfg = {
        "azure2_exe": "AZURE2",
        "use_brune": True,
        "use_gsl": True,
        "n_samples": 100,
        "max_workers": 4,
        "seed": 42,
        "keep_tmp": False,
        "timeout": 600,
        "output_file": "mc_results.npz",
        "params_file": params_out,
        "quantiles": [0.16, 0.50, 0.84],
        # "output_prefix": "mc_results",  # Optional: prefix for .dat files
    }

    with open(setup_out, "w") as fh:
        fh.write("# AZURE2 Monte Carlo — run setup\n")
        fh.write("# --------------------------------\n")
        fh.write(f"# Generated from: {azr_filepath}\n")
        fh.write("#\n")
        fh.write("# To run:\n")
        fh.write(f"#   python azure_mc.py run {azr_filepath} {setup_out}\n")
        fh.write("#\n")
        fh.write("# 'quantiles' are used for both the MC run and can be reused with:\n")
        fh.write(f"#   python azure_mc.py quantiles mc_results.npz -c {setup_out}\n\n")
        yaml.dump(setup_cfg, fh, default_flow_style=False, sort_keys=False)

    print(f"Setup  written to {setup_out}")
    print(f"Params written to {params_out}")
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


def cmd_run(
    azr_filepath: str,
    setup_filepath: str,
    tmp_dir: str | None = None,
):
    """Run the Monte Carlo, collect distributions, compute quantiles."""
    with open(setup_filepath, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    n_samples = cfg.get("n_samples", 100)
    max_workers = cfg.get("max_workers", 4)
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
    timeout = cfg.get("timeout", 600)
    output_file = cfg.get("output_file", "mc_results.npz")
    quantiles_list = cfg.get("quantiles", [0.16, 0.50, 0.84])

    # Load parameters from separate file
    params_filepath = cfg.get("params_file", "mc_params.yaml")
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

    ranges: list[dict] = []
    for i, key in enumerate(all_keys):
        nom = nominals[i]
        if key in user_params:
            r = dict(user_params[key])
            if "low" not in r or "high" not in r:
                half = abs(nom) * default_frac if nom != 0 else 1.0
                r.setdefault("low", nom - half)
                r.setdefault("high", nom + half)
            r.setdefault("distribution", default_dist)
        else:
            half = abs(nom) * default_frac if nom != 0 else 1.0
            r = {"low": nom - half, "high": nom + half,
                 "distribution": default_dist}
        ranges.append(r)

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
                    for pending in futures:
                        pending.cancel()
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
        save_dict[f"{safe_ch}/quantiles_xs"] = q_xs
        save_dict[f"{safe_ch}/quantiles_sf"] = q_sf

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

    np.savez(output_file, **save_dict)
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
        key_qx = f"{safe_ch}/quantiles_xs"
        key_qs = f"{safe_ch}/quantiles_sf"
        if key_e not in data:
            continue

        energies = data[key_e]
        bucket_xs = data[key_bx]
        q_xs = data[key_qx]
        q_sf = data[key_qs]

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
            print()
        print()

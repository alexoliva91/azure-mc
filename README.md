# azure-mc

Monte Carlo error estimation for AZURE2 R-matrix calculations.

Supports two modes of operation:

1. **Random MC (extrapolation)** — samples parameters within
   user-defined ranges and runs AZURE2 in *Extrapolate Without Data* mode.
2. **MCMC (fit to data)** — uses the [emcee](https://emcee.readthedocs.io/)
   ensemble sampler to explore the posterior distribution of R-matrix
   parameters by evaluating the χ² likelihood via AZURE2's
   *Calculate With Data* mode.  The parameter ranges in `params.yaml`
   serve as priors.

Each mode has its own `populate` subcommand that generates a dedicated
setup file (`mc_setup.yaml` or `mcmc_setup.yaml`) plus a shared
`params.yaml` for the free-parameter definitions.

The classes and functions in this package are inspired by those of
the [BRICK](https://github.com/odell/brick) toolkit.

## Requirements

- Python 3.9+
- NumPy
- SciPy (for distributions beyond uniform/gaussian)
- PyYAML
- emcee (for MCMC mode)
- AZURE2 executable in PATH

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
azure_mc.py              # Entry point (backwards-compatible script)
azure_mc/                # Package
├── __init__.py          # Package init + logging config
├── __main__.py          # Enables `python -m azure_mc`
├── constants.py         # .azr column indices
├── models.py            # Level, Parameter, NormFactor classes
├── io.py                # Read/write .azr files, parse .extrap output
├── parameters.py        # Parameter discovery, extraction, MC sampling
├── runner.py            # AZURE2 execution, single MC run logic
├── commands.py          # Subcommand implementations
└── cli.py               # Argparse CLI entry point
```

## Usage

```bash
python -m azure_mc <command> <subcommand> [options]
```

For backwards compatibility, `python azure_mc.py ...` also works.

Commands:

| Command | Description |
|---------|-------------|
| `mc populate` | Discover free parameters → `mc_setup.yaml` + `params.yaml` |
| `mc extrapolate` | Sample parameters randomly and extrapolate |
| `mc summary` | Inspect random-MC results |
| `mcmc populate` | Discover free parameters → `mcmc_setup.yaml` + `params.yaml` |
| `mcmc fit` | Fit parameters to data via MCMC |
| `mcmc extrapolate` | Extrapolate using posterior samples |
| `mcmc summary` | Inspect MCMC / extrapolation results |

### Workflow A — Random MC (extrapolation without data)

#### Step 1: Discover parameters

```bash
python -m azure_mc mc populate -i input.azr
```

This creates two files:
- `mc_setup.yaml` — MC run configuration (n_samples, workers, quantiles, …)
- `params.yaml` — per-parameter ranges and distributions (shared with MCMC)

#### Step 2: Edit parameter ranges

Edit `params.yaml` to adjust the sampling ranges, distributions, and
sigma values for each free parameter.

#### Step 3: Extrapolate

```bash
python -m azure_mc mc extrapolate -i input.azr -c mc_setup.yaml
```

#### Step 4: Inspect results

```bash
python -m azure_mc mc summary -r mc_extrapolate.npz
```

#### Step 5 (optional): Recompute quantiles

Pass `-q` to `summary` to also write `.dat` files:

```bash
python -m azure_mc mc summary -r mc_extrapolate.npz -q 0.025 0.5 0.975

# With a custom output file prefix:
python -m azure_mc mc summary -r mc_extrapolate.npz -q 0.16 0.84 -p my_prefix
```

### Workflow B — MCMC (fit to experimental data)

This mode uses [emcee](https://emcee.readthedocs.io/) to perform a Markov
Chain Monte Carlo exploration of the parameter posterior, using AZURE2's
*Calculate With Data* mode to evaluate the χ² likelihood at each proposed
parameter vector.  The ranges/distributions in `params.yaml` serve as
**priors** (uniform → flat prior within bounds; gaussian → Gaussian prior
centered on `nominal` with width `sigma`, hard-bounded by `low`/`high`).

#### Step 1: Discover parameters

```bash
python -m azure_mc mcmc populate -i input.azr
```

This creates two files:
- `mcmc_setup.yaml` — MCMC settings (n_walkers, n_steps, …)
- `params.yaml` — per-parameter ranges / priors (shared with MC)

#### Step 2: Edit priors

Edit `params.yaml`.  For MCMC the `low`/`high` bounds are hard prior
boundaries.  Use `distribution: gaussian` with a `sigma` key for
informative priors.

#### Step 3: (Optional) Tune MCMC settings

In `mcmc_setup.yaml`, edit:

| Key | Default | Description |
|-----|---------|-------------|
| `n_walkers` | `2*N_params + 2` | Number of emcee walkers (must be ≥ 2 × n_params) |
| `n_steps` | `1000` | Total MCMC steps per walker |
| `n_burn` | `200` | Burn-in steps to discard |
| `thin` | `1` | Thinning factor |
| `init_spread` | `1e-4` | Initial ball spread (fraction of range) |
| `output_file` | `mcmc_chain.npz` | Chain output file |
| `extrapolate_output_file` | `mcmc_extrapolate.npz` | Posterior extrapolation output |
| `progress` | `true` | Show progress bar (requires tqdm) |

#### Step 4: Fit

```bash
python -m azure_mc mcmc fit -i input.azr -c mcmc_setup.yaml
```

The output `mcmc_chain.npz` contains:
- `chain` — full chain `(n_steps, n_walkers, n_params)`
- `flat_chain` — flattened chain after burn-in/thinning
- `log_prob` — log-posterior values
- `acceptance_fraction`, `autocorr_time`
- `posterior_quantiles` — parameter quantiles

#### Step 5: Posterior extrapolation

Draw samples from the posterior chain and run AZURE2 in extrapolation mode
to produce uncertainty bands on the cross section / S-factor:

```bash
python -m azure_mc mcmc extrapolate -i input.azr -c mcmc_setup.yaml --chain mcmc_chain.npz
# or with a specific number of draws:
python -m azure_mc mcmc extrapolate -i input.azr -c mcmc_setup.yaml --chain mcmc_chain.npz -n 200
```

## Configuration

### mc_setup.yaml

Generated by `mc populate`. Contains only MC-relevant settings:

| Key | Default | Description |
|-----|---------|-------------|
| `azure2_exe` | `AZURE2` | Path to AZURE2 executable |
| `use_brune` | `true` | Use Brune parameterization |
| `use_gsl` | `true` | Use GSL Coulomb functions |
| `n_samples` | `100` | Number of MC samples |
| `max_workers` | `4` | Parallel worker count |
| `seed` | `42` | Random seed for reproducibility |
| `keep_tmp` | `false` | Keep temporary run directories |
| `timeout` | `600` | Per-run timeout in seconds |
| `output_file` | `mc_extrapolate.npz` | Output file path |
| `params_file` | `params.yaml` | Parameters file path |
| `quantiles` | `[0.16, 0.50, 0.84]` | Quantile levels to compute |

### mcmc_setup.yaml

Generated by `mcmc populate`. Contains only MCMC-relevant settings
in a flat structure (no nested blocks):

| Key | Default | Description |
|-----|---------|-------------|
| `azure2_exe` | `AZURE2` | Path to AZURE2 executable |
| `use_brune` | `true` | Use Brune parameterization |
| `use_gsl` | `true` | Use GSL Coulomb functions |
| `max_workers` | `4` | Parallel worker count |
| `seed` | `42` | Random seed for reproducibility |
| `timeout` | `600` | Per-run timeout in seconds |
| `params_file` | `params.yaml` | Parameters file path |
| `quantiles` | `[0.16, 0.50, 0.84]` | Quantile levels to compute |
| `n_walkers` | `2*N+2` | Number of emcee walkers |
| `n_steps` | `1000` | MCMC steps per walker |
| `n_burn` | `200` | Burn-in steps to discard |
| `thin` | `1` | Thinning factor |
| `init_spread` | `1e-4` | Walker initialisation spread |
| `output_file` | `mcmc_chain.npz` | Chain output file |
| `extrapolate_output_file` | `mcmc_extrapolate.npz` | Extrapolation output file |
| `progress` | `true` | Show progress bar |

### params.yaml

Generated by either `mc populate` or `mcmc populate` — the file is
identical regardless of which command creates it.  Both setup files
point to it via the `params_file` key.

- For **MC**: `low`/`high` act as hard sampling bounds (clipping).
- For **MCMC**: `low`/`high` act as hard prior bounds (−∞ outside).

Global defaults can be set under the `defaults` key:

| Key | Default | Description |
|-----|---------|-------------|
| `fraction` | `0.2` | Default ±fraction for parameters missing explicit bounds |
| `distribution` | `uniform` | Default distribution type |

Each parameter entry supports:

| Key | Description |
|-----|-------------|
| `nominal` | Central value (from .azr file) |
| `low` | Lower bound (hard clipping / prior truncation) |
| `high` | Upper bound (hard clipping / prior truncation) |
| `distribution` | Distribution name (see below) |
| `sigma` | Std deviation (for `gaussian` and `lognormal`) |
| `mu` | Mean of ln(X) (for `lognormal`; default: `ln(|nominal|)`) |
| `dist_params` | Dict of shape/loc/scale params (for scipy distributions) |

#### Built-in distributions

| Name | Parameters | Description |
|------|-----------|-------------|
| `uniform` | `low`, `high` | Flat between bounds (default) |
| `gaussian` | `sigma` | Normal centered on `nominal`; default σ = (high−low)/4 |
| `lognormal` | `mu`, `sigma` | Log-normal; defaults: mu = ln(|nominal|), σ = 1 |

#### scipy.stats distributions

Any [scipy.stats continuous distribution](https://docs.scipy.org/doc/scipy/reference/stats.html)
can be used by setting `distribution` to the scipy name and providing
shape/loc/scale parameters under `dist_params`.  `low`/`high` always
act as hard bounds on top of the distribution.

Examples:

```yaml
# Truncated normal (hard bounds from low/high)
distribution: truncnorm
dist_params:
  a: -2       # lower clip in std devs
  b: 2        # upper clip in std devs
  loc: 1.0    # mean
  scale: 0.5  # std dev

# Gamma distribution
distribution: gamma
dist_params:
  a: 2.0
  scale: 50.0

# Beta distribution
distribution: beta
dist_params:
  a: 2.0
  b: 5.0
  loc: 10.0
  scale: 5.0
```

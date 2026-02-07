# azure-mc

Monte Carlo error estimation for AZURE2 R-matrix calculations.

The classes and functions in this package are inspired by those of
the [BRICK](https://github.com/odell/brick) toolkit. 

## Requirements

- Python 3.9+
- NumPy
- PyYAML
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

The tool can be invoked in two ways:

```bash
python azure_mc.py <command> ...
# or
python -m azure_mc <command> ...
```

### Step 1: Discover parameters

```bash
python azure_mc.py populate input.azr
```

This creates two files:
- `mc_setup.yaml` — run configuration (n_samples, workers, quantiles, etc.)
- `mc_params.yaml` — per-parameter ranges and distributions

### Step 2: Edit parameter ranges

Edit `mc_params.yaml` to adjust the sampling ranges, distributions, and
sigma values for each free parameter.

### Step 3: Run Monte Carlo

```bash
python azure_mc.py run input.azr mc_setup.yaml
```

### Step 4: Inspect results

```bash
python azure_mc.py summary mc_results.npz
```

### Step 5 (optional): Recompute quantiles

Generate `.dat` files for different quantile levels without re-running the MC:

```bash
# Specify quantiles directly:
python azure_mc.py quantiles mc_results.npz -q 0.025 0.5 0.975

# Or read quantiles from the setup YAML:
python azure_mc.py quantiles mc_results.npz -c mc_setup.yaml

# Optionally set a custom output file prefix:
python azure_mc.py quantiles mc_results.npz -q 0.16 0.84 -p my_prefix
```

## Configuration

### mc_setup.yaml

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
| `output_file` | `mc_results.npz` | Output file path |
| `params_file` | `mc_params.yaml` | Parameters file path |
| `quantiles` | `[0.16, 0.50, 0.84]` | Quantile levels to compute |

### mc_params.yaml

Global defaults can be set under the `defaults` key:

| Key | Default | Description |
|-----|---------|-------------|
| `fraction` | `0.2` | Default ±fraction for parameters missing explicit bounds |
| `distribution` | `uniform` | Default distribution type |

Each parameter entry supports:

| Key | Description |
|-----|-------------|
| `nominal` | Central value (from .azr file) |
| `low` | Lower bound |
| `high` | Upper bound |
| `distribution` | `uniform` or `gaussian` |
| `sigma` | Standard deviation (gaussian only; default: `(high-low)/4`) |

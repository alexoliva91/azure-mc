# azure-mc

Monte Carlo error estimation for AZURE2 R-matrix calculations.

The classes and functions in this package are inspired by those of
the [BRICK](https://github.com/odell/brick) toolkit. 

## Requirements

- Python 3.9+
- NumPy
- PyYAML
- AZURE2 executable in PATH (`engine: legacy`), and/or
- AZURE2 dev branch build + its `pyazr` package importable (`engine: pyazr`,
  the default — see [Execution engines](#execution-engines) below)

## Installation

```bash
pip install -r requirements.txt
```

`pyazr` is not on PyPI — it ships as a plain package directory inside an
AZURE2 dev-branch checkout (`<repo>/pyazr`) and must be built/obtained
separately. See [Execution engines](#execution-engines).

## Project Structure

```
azure_mc.py              # Entry point (backwards-compatible script)
azure_mc/                # Package
├── __init__.py          # Package init + logging config
├── __main__.py          # Enables `python -m azure_mc`
├── constants.py         # .azr column indices (legacy engine)
├── models.py            # Level, Parameter, NormFactor classes (legacy engine)
├── io.py                # Read/write .azr files, parse .extrap output (legacy engine)
├── parameters.py        # Parameter discovery, extraction, MC sampling
├── runner.py            # AZURE2 subprocess execution (legacy engine)
├── pyazr_adapter.py      # pyazr session/parameter/calculate glue (pyazr engine)
├── commands.py          # Subcommand implementations (branches on 'engine')
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

## Execution engines

azure_mc supports two interchangeable execution backends, selected via the
`engine` key in `mc_setup.yaml` (or the `--engine` CLI flag on `populate`/`run`):

- **`pyazr`** (default) — opens one persistent AZURE2 API session (via the
  AZURE2 dev branch's socket-based `pyazr` package) and dispatches every MC
  sample against it over a thread pool. No per-sample process spawn or file
  I/O, so it's substantially faster than the legacy engine. Requires an
  AZURE2 dev-branch build and its `pyazr` package to be importable — point
  `azure2_binary` / `pyazr_path` at your build, or add `pyazr`'s parent
  directory to `PYTHONPATH` yourself. `populate` automatically falls back to
  `legacy` (with a warning) if `pyazr` can't be imported.
- **`legacy`** — spawns one `AZURE2 --no-gui` subprocess per sample,
  rewriting a temporary `.azr` file each time (azure_mc's original
  behavior). Only needs a plain AZURE2 executable on `PATH`.

Parameter keys discovered under each engine differ (they come from
different discovery mechanisms), so **switching a model's `engine` requires
re-running `populate`** to regenerate `mc_params.yaml`.

## Configuration

### mc_setup.yaml

| Key | Default | Description |
|-----|---------|-------------|
| `engine` | `pyazr` | Execution backend: `pyazr` or `legacy` |
| `azure2_binary` | *(unset)* | Path to the AZURE2 dev-branch binary (pyazr engine) |
| `pyazr_path` | *(unset)* | Directory containing the `pyazr` package, if not already importable (pyazr engine) |
| `azure2_exe` | `AZURE2` | Path to AZURE2 executable (legacy engine) |
| `use_brune` | `true` | Use Brune parameterization (legacy engine) |
| `use_gsl` | `true` | Use GSL Coulomb functions (legacy engine) |
| `n_samples` | `100` | Number of MC samples |
| `max_workers` | `4` | Parallel worker count (thread pool for pyazr, process pool for legacy) |
| `seed` | `42` | Random seed for reproducibility |
| `keep_tmp` | `false` | Keep temporary run directories (legacy engine) |
| `timeout` | `1800` (pyazr) / `600` (legacy) | Per-run / API-call timeout in seconds |
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

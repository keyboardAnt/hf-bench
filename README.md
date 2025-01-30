# hf-bench

## Setup

```bash
conda env create -f conda/environment.yml
```

Include an `.env` file in the root directory with the following variables:
```
HF_ACCESS_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_HOME=/path/to/hf_home
```

## CLI Usage

The package provides several command-line interfaces:

### Cluster Commands

Generate cluster submission commands:
```bash
# Generate and preview submission command
python -m hf_bench cluster get-command -c config.yaml --dry-run

# Generate and execute submission command (with confirmation prompt)
python -m hf_bench cluster get-command -c config.yaml
```

### Benchmark Commands

Run and monitor benchmarks:
```bash
# Run benchmark with monitoring and result analysis
python -m hf_bench benchmark run -c config.yaml --monitor --analyze

# Run benchmark without monitoring
python -m hf_bench benchmark run -c config.yaml

# Check status of a running benchmark
python -m hf_bench benchmark status -c config.yaml JOB_ID
```

All commands require a config file specified with `-c` or `--config`.

## Tests

```bash
pytest
```

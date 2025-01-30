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

## Tests

```bash
pytest
```

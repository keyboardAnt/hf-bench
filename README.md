# hf-bench

## Setup

Create a new conda environment:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate hf-bench-env
```

Please include an `.env` file in the root directory with the following variables. The models and datasets are downloaded to the `HF_HOME` directory, unless they are already stored there.
```
HF_ACCESS_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_HOME=/path/to/hf_home
```

To update the environment after changing the `environment.yml` file:
```bash
conda env update -f environment.yml --prune
```

## CLI Usage

### Benchmark

> [!NOTE] To run the benchmark, you must login to Weights & Biases and often also Hugging Face.

Below are commands for running the benchmark on 
1. Machine with GPUs

    ```bash
    python -m hf_bench.benchmark --num_of_examples=1
    ```

2. Cluster

    ```bash
    ./hf_bench/submit/lsf.sh --num_of_examples=1
    ```

After running the above sanity check with one example and the `default` experiment config, you can run the benchmark with 30 examples and a custom experiment config:

```bash
python -m hf_bench.benchmark --experiment_config deepseek-r1-qwen-32b
```
or
```bash
./hf_bench/submit/lsf.sh --experiment_config deepseek-r1-qwen-32b
```

To adjust the hardware request to the cluster, edit the submit script.

#### Monitoring

After loading the models, you can monitor the progress of the benchmark here: https://wandb.ai/generating-faster/hf-bench.

<!-- ## Tests
Run tests in parallel:
```bash
pytest -n 4 -v
``` -->

### Analyze

Analyze the CSV of benchmark results:
```bash
python -m hf_bench.analyze --csv_path=path/to/csv
```

Summarize the benchmark results:
```bash
python -m  hf_bench.summarize_results --dirpath benchmark_results
```
This will create new CSV files in the `results` branch:
* [`results_all.csv`](https://github.com/keyboardAnt/hf-bench/blob/results/results_all.csv)
* [`results_summary.csv`](https://github.com/keyboardAnt/hf-bench/blob/results/results_summary.csv)
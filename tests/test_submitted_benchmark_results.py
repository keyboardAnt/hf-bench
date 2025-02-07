from collections import Counter
import warnings
import pandas as pd
from hf_bench.summarize_results import list_tracked_files, list_staged_files, get_columns


def test_submitted_benchmark_results():
    """Test that the submitted benchmark results follow the correct format."""
    filepaths = list_tracked_files("benchmark_results")
    filepaths += list_staged_files("benchmark_results")
    expected_columns = get_columns()
    print(expected_columns)
    benign_missing_example_ids_compared_to_others = [] # Some (target, dataset_path, drafter, temperature) have less unique example IDs than others
    benign_missing_example_ids_compared_to_declared = [] # Some (target, dataset_path, drafter, temperature) have less unique example IDs than declared in the num_of_examples column
    catastrophic_missing_example_ids_compared_to_others = [] # Some (target, dataset_path, drafter, temperature, example_id) include less rows than excepted, where the expected number of rows is the Cartesian product of the number of unique (target, dataset_path, drafter, temperature)
    catastrophic_non_positive_values = [] # Some values are non-positive
    for f in filepaths:
        df = pd.read_csv(f)
        col_counter = Counter(df.columns)
        for col in expected_columns:
            assert col_counter[col] == 1, f"CATASTROPHIC: Column {col} is missing in the dataframe or appears multiple times.\nFilepath: {f}"
        # Check that all example IDs appear the same number of times
        columns_for_index = ["target", "dataset_path", "drafter", "temperature"]
        df_example_ids_nunique = df.groupby(columns_for_index)["example_id"].nunique()
        if df_example_ids_nunique.min() != df_example_ids_nunique.max():
            print(f"BENIGN: File {f} has missing example IDs (example IDs do not appear the same number of times).")
            benign_missing_example_ids_compared_to_others.append(f)
        # Check that all example IDs appear num_of_examples times
        num_of_examples_max = df["num_of_examples"].max()
        if df_example_ids_nunique.min() != num_of_examples_max:
            print(f"BENIGN: File {f} has wrong example IDs that repeat less than the declared number of times, which is num_of_examples={num_of_examples_max}.")
            benign_missing_example_ids_compared_to_declared.append(f)
        # Calculate the expected number of times each example ID should appear
        # This is the Cartesian product of the number of unique (target, dataset_path, drafter, temperature)
        df["drafter"] = df["drafter"].fillna("No Drafter (Autoregressive)")
        expected_num_of_rows_per_example_id: int = df["target"].nunique() * df["dataset_path"].nunique() * df["drafter"].nunique() * df["temperature"].nunique()
        # When grouping by (target, dataset_path, drafter, temperature, example_id), the number of rows should be the expected number of rows per example ID
        df_example_ids_count = df[columns_for_index + ["example_id"]].groupby("example_id").count().min(axis=1)
        mask_positive = df_example_ids_count < expected_num_of_rows_per_example_id
        catastrophic_missing_example_ids = df_example_ids_count[mask_positive].index.get_level_values("example_id")
        if len(catastrophic_missing_example_ids) > 0:
            print(f"CATASTROPHIC: File {f} has catastrophic missing example IDs. The following example IDs do not repeat the expected number of times, which is {expected_num_of_rows_per_example_id}={df['target'].nunique()}*{df['dataset_path'].nunique()}*{df['drafter'].nunique()}*{df['temperature'].nunique()}:\n{catastrophic_missing_example_ids.to_list()}")
            catastrophic_missing_example_ids_compared_to_others.append(f)
        # Check that all the values are strictly positive
        mask_positive = df[["new_toks", "ttft_ms", "tpot_ms", "out_toks_per_sec"]] > 0
        if not mask_positive.all().all().item():
            print(f"CATASTROPHIC: File {f} has non-positive values.")
            catastrophic_non_positive_values.append(f)
    if benign_missing_example_ids_compared_to_others:
        warnings.warn("BENIGN: Some example IDs do not appear the same number of times in the following files:\n" + " ".join(benign_missing_example_ids_compared_to_others))
    if benign_missing_example_ids_compared_to_declared:
        warnings.warn("BENIGN: Some example IDs appear only %d times in the dataframe although they should appear %d times according to the num_of_examples column.\nFilepath: %s" % (df_example_ids_nunique.min(), num_of_examples_max, f))
    assert len(catastrophic_missing_example_ids_compared_to_others) == 0, f"CATASTROPHIC: Some example IDs compare only a proper subset of the methods in the following files:\n" + " ".join(catastrophic_missing_example_ids_compared_to_others)
    assert len(catastrophic_non_positive_values) == 0, f"CATASTROPHIC: Some values are non-positive in the following files:\n" + " ".join(catastrophic_non_positive_values)
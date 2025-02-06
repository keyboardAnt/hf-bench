from collections import Counter
import warnings
import pandas as pd
from hf_bench.summarize_results import list_tracked_files, get_columns


def test_submitted_benchmark_results():
    """Test that the submitted benchmark results follow the correct format."""
    filepaths = list_tracked_files("benchmark_results")
    expected_columns = get_columns()
    print(expected_columns)
    for f in filepaths:
        df = pd.read_csv(f)
        col_counter = Counter(df.columns)
        for col in expected_columns:
            assert col_counter[col] == 1, f"Column {col} is missing in the dataframe or appears multiple times.\nFilepath: {f}"
        # Check that all example IDs appear the same number of times
        columns_for_index = ["target", "dataset_path", "drafter", "temperature"]
        df_example_ids_nunique = df.groupby(columns_for_index)["example_id"].nunique()
        assert df_example_ids_nunique.min() == df_example_ids_nunique.max(), f"All example IDs should appear the same number of times.\nFilepath: {f}"
        # Check that all example IDs appear num_of_examples times
        expected_count = df["num_of_examples"].max()
        if df_example_ids_nunique.min() != expected_count:
            msg: str = f"Some example IDs appear only {df_example_ids_nunique.min()} times in the dataframe although they should appear {expected_count} times according to the num_of_examples column.\nFilepath: {f}"
            warnings.warn(msg)


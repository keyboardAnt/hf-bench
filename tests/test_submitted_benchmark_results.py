from collections import Counter
import pandas as pd
from hf_bench.parse_results import list_tracked_files, get_columns


def test_submitted_benchmark_results():
    """Test that the submitted benchmark results follow the correct format."""
    filepaths = list_tracked_files("benchmark_results")
    expected_columns = get_columns()
    print(expected_columns)
    for f in filepaths:
        df = pd.read_csv(f)
        col_counter = Counter(df.columns)
        for col in expected_columns:
            assert col_counter[col] == 1, f"Column {col} is missing in the dataframe or appears multiple times"
        # Check all the example_ids are repeated the same number of times
        example_ids_counts = df["example_id"].value_counts()
        for example_id, count in example_ids_counts.items():
            assert count == example_ids_counts[0], f"All example IDs should appear the same number of times.\nExample ID {example_id} appears {count} times in the dataframe.\nValue counts: {example_ids_counts=}\nFilepath: {f}"

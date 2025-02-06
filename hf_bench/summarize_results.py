from pathlib import Path
import subprocess
from typing import Any, Dict, List

import fire
import pandas as pd
from scipy.stats import hmean
from hf_bench.benchmark import ResultsTableRow


def get_columns() -> List[str]:
    return ResultsTableRow.__annotations__.keys()


def list_tracked_files(dirpath: str) -> Dict[str, str]:
    # Run git ls-tree command and capture output
    cmd = ["git", "ls-tree", "-r", "HEAD", "--name-only", dirpath]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Split output into list of files
    files = result.stdout.strip().split("\n")
    # Filter out empty strings
    files = [f for f in files if f]
    return files


def get_df_concat(dirpath: str) -> pd.DataFrame:
    """
    Get a dataframe of all the results in the given directory.
    """
    filepaths = list_tracked_files(dirpath)
    print(f"Found {len(filepaths)} tracked files in {dirpath}.")
    df_first = pd.read_csv(filepaths[0])
    column_dtypes: Dict[str, Any] = {
        "submission_id": str,
        **{col: df_first[col].dtype for col in get_columns()},
    }
    columns = ["submission_id"] + list(get_columns())
    df = pd.DataFrame(columns=columns).astype(column_dtypes)
    for f in filepaths:
        submission_id: str = Path(f).parent.stem
        df_new = pd.read_csv(f)
        df_new["drafter"] = df_new["drafter"].fillna("No Drafter (Autoregressive)")
        df_new["submission_id"] = submission_id
        df_new = df_new[columns]
        df = pd.concat([df, df_new])
    df.sort_values(
        by=columns,
        inplace=True,
    )
    return df


def get_df_summary_of_results(df_concat: pd.DataFrame) -> pd.DataFrame:
    df_concat.reset_index(drop=True, inplace=True)
    columns_for_index: List[str] = [
        "target",
        "submission_id",
        "dataset_path",
        "drafter",
        "temperature",
    ]
    df_concat.set_index(columns_for_index, inplace=True)
    example_id_nunique = df_concat["example_id"].groupby(columns_for_index).nunique()
    df_summary = example_id_nunique.to_frame()
    df_summary.rename(columns={"example_id": "example_id_nunique"}, inplace=True)
    df_mean_vals = df_concat.groupby(columns_for_index)[["new_toks", "ttft_ms"]].mean()
    df_mean_vals['new_toks'] = df_mean_vals['new_toks'].round(0).astype(int)
    df_mean_vals['ttft_ms'] = df_mean_vals['ttft_ms'].round(1)
    df_hmean_vals = df_concat.groupby(columns_for_index)[
        ["tpot_ms", "out_toks_per_sec"]
    ].agg(hmean).round(1)

    # Calculate speedup
    # Reset index to bring drafter back as a column
    df_concat.reset_index(inplace=True)
    # Filter out rows where drafter is 'No Drafter (Autoregressive)'
    reference_df = df_concat[df_concat['drafter'] == 'No Drafter (Autoregressive)']
    # Group by the columns that will match across rows and get the first value of out_toks_per_sec
    reference_df = reference_df.groupby(['target', 'submission_id', 'dataset_path', 'temperature'], as_index=False)['out_toks_per_sec'].first()
    # Merge reference_df back to the original DataFrame based on matching group keys
    df_concat = df_concat.merge(reference_df, on=['target', 'submission_id', 'dataset_path', 'temperature'], suffixes=('', '_ref'))
    # Add a new column 'speedup' by dividing 'out_toks_per_sec' by the reference value
    df_concat['speedup'] = (df_concat['out_toks_per_sec'] / df_concat['out_toks_per_sec_ref']).round(2)
    # Drop the reference column as it's no longer needed
    df_concat.drop(columns=['out_toks_per_sec_ref'], inplace=True)
    # Set the index back to the original columns
    df_concat.set_index(columns_for_index, inplace=True)
    # Merge df_concat with df_summary on the group keys to bring speedup into the summary dataframe
    df_summary = pd.concat([df_summary, df_mean_vals, df_hmean_vals], axis=1)
    # Merge the speedup column into df_summary by matching the index
    df_summary = df_summary.merge(df_concat[['speedup']], left_index=True, right_index=True, how='left')

    return df_summary

def main(dirpath: str):
    print("Concatenating all the results CSVs into one dataframe...")
    df_concat: pd.DataFrame = get_df_concat(dirpath)
    df_concat.to_csv("results_all.csv", index=False)
    print("Counting the number of unique example IDs for each experiment...")
    df_summary: pd.DataFrame = get_df_summary_of_results(df_concat)
    df_summary.to_csv("results_summary.csv", index=True)
    print(f"Stored both the concatenated dataframe and the summary in {dirpath}.")
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)

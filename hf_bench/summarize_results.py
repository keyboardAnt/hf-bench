from pathlib import Path
import subprocess
from typing import Any, Dict, List

import fire
import pandas as pd
from scipy.stats import hmean
from hf_bench.benchmark import ResultsTableRow


def get_columns() -> List[str]:
    return ResultsTableRow.__annotations__.keys()


def list_tracked_files(dirpath: str) -> List[str]:
    # Run git ls-tree command and capture output
    cmd = ["git", "ls-tree", "-r", "HEAD", "--name-only", dirpath]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Split output into list of files
    files = result.stdout.strip().split("\n")
    # Filter out empty strings
    files = [f for f in files if f]
    return files


def list_staged_files(dirpath: str) -> List[str]:
    cmd = ["git", "diff", "--name-only", "--cached", "HEAD", dirpath]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = result.stdout.strip().split("\n")
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
    df_hmean_vals = df_concat.groupby(columns_for_index)[
        ["tpot_ms", "out_toks_per_sec"]
    ].agg(hmean)
    df_summary = pd.concat([df_summary, df_mean_vals, df_hmean_vals], axis=1)
    return df_summary


def main(dirpath: str):
    print("Concatenating all the results CSVs into one dataframe...")
    df_concat: pd.DataFrame = get_df_concat(dirpath)
    df_concat.to_csv("results_all.csv", index=False)

    print("Counting the number of unique example IDs for each experiment...")
    df_summary: pd.DataFrame = get_df_summary_of_results(df_concat)
    # Round the values to 1 decimal place
    df_summary["new_toks"] = df_summary["new_toks"].round(1)
    df_summary["ttft_ms"] = df_summary["ttft_ms"].round(1)
    df_summary["tpot_ms"] = df_summary["tpot_ms"].round(1)
    df_summary["out_toks_per_sec"] = df_summary["out_toks_per_sec"].round(1)
    # Reorder the multi-index columns so that the `submission_id` column is the last one
    df_summary.reset_index(level="submission_id", inplace=True)
    df_summary.set_index("submission_id", append=True, inplace=True)
    df_summary.to_csv("results_summary.csv", index=True)

    print(f"Stored both the concatenated dataframe and the summary in {dirpath}.")
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)

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


def get_df_concat_filtered(df_concat: pd.DataFrame, minimum_new_toks: int) -> pd.DataFrame:
    df_concat_filtered = df_concat.set_index(["target", "submission_id", "dataset_path", "dataset_name", "dataset_split"])
    df_low_new_toks = (df_concat_filtered[df_concat_filtered["new_toks"] < minimum_new_toks]
                   .set_index("example_id", append=True)
                   .sort_index())
    index_low_new_toks = df_low_new_toks.index.unique() # Multi-index (target, submission_id, dataset_path, dataset_name, dataset_split, example_id) for which new_toks < 64
    # Remove all the rows corresponding to these multi-indices from df_concat_filtered
    df_concat_filtered.set_index("example_id", inplace=True, append=True)
    df_concat_filtered = df_concat_filtered[~df_concat_filtered.index.isin(index_low_new_toks)]
    df_concat_filtered.reset_index(inplace=True)
    return df_concat_filtered


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
    # Add the speedups
    df_otps = df_summary[["out_toks_per_sec"]]
    df_otps.reset_index(level="drafter", inplace=True)
    mask_ar = df_otps["drafter"] == "No Drafter (Autoregressive)"
    df_ar_otps = df_otps[mask_ar]
    df_ar_otps.drop(columns=["drafter"], inplace=True)
    # Reset the index of both dataframes to make the division operation simpler
    df_otps_reset = df_otps.reset_index()
    df_ar_otps_reset = df_ar_otps.reset_index()
    # Merge the dataframes on the common index columns
    merge_cols = ["target", "dataset_path", "temperature", "submission_id"]
    df_merged = pd.merge(
        df_otps_reset, df_ar_otps_reset, on=merge_cols, suffixes=("", "_ar")
    )
    # Perform the division
    df_merged["speedup"] = (
        df_merged["out_toks_per_sec"] / df_merged["out_toks_per_sec_ar"]
    )
    # Set back the multi-index structure
    df_speedups = df_merged.set_index(merge_cols + ["drafter"])[["speedup"]]
    df_summary.reset_index(inplace=True)
    df_summary.set_index(
        ["target", "dataset_path", "drafter", "temperature", "submission_id"],
        inplace=True,
    )
    df_summary = df_summary.join(df_speedups)
    # Reorder the multi-index columns
    df_summary.reset_index(inplace=True)
    new_index = ["target", "dataset_path", "submission_id", "temperature", "drafter"]
    df_summary.set_index(new_index, inplace=True)
    df_summary.sort_index(level=new_index, inplace=True)
    return df_summary


def get_df_max_speedup(df_summary: pd.DataFrame) -> pd.DataFrame:
    df_summary.reset_index(inplace=True)
    df_max_speedup = df_summary.loc[
        df_summary.groupby(["target", "dataset_path", "submission_id", "temperature"])[
            "speedup"
        ].idxmax()
    ]
    df_max_speedup.rename(columns={"drafter": "drafter_of_max_speedup"}, inplace=True)
    df_max_speedup.set_index(
        [
            "target",
            "temperature",
            "dataset_path",
            "submission_id",
            "drafter_of_max_speedup",
        ],
        inplace=True,
    )
    df_max_speedup.sort_index(inplace=True)
    return df_max_speedup


def main(dirpath: str):
    print("Concatenating all the results CSVs into one dataframe...")
    df_concat: pd.DataFrame = get_df_concat(dirpath)
    df_concat.to_csv("results_all.csv", index=False)

    minimum_new_toks = 128
    print(f"Filtering out experiments with less than {minimum_new_toks} new tokens...")
    df_concat_filtered = get_df_concat_filtered(df_concat, minimum_new_toks)

    print("Counting the number of unique example IDs for each experiment...")
    df_summary: pd.DataFrame = get_df_summary_of_results(df_concat_filtered)
    # Round the values to 1 decimal place
    df_summary["new_toks"] = df_summary["new_toks"].round(1)
    df_summary["ttft_ms"] = df_summary["ttft_ms"].round(1)
    df_summary["tpot_ms"] = df_summary["tpot_ms"].round(1)
    df_summary["out_toks_per_sec"] = df_summary["out_toks_per_sec"].round(1)
    df_summary["speedup"] = df_summary["speedup"].round(2)
    df_summary.to_csv("results_summary.csv", index=True)

    print("Getting the maximum speedup for each experiment...")
    df_max_speedup: pd.DataFrame = get_df_max_speedup(df_summary.copy())
    df_max_speedup.to_csv("results_max_speedup.csv", index=True)

    print("Getting the summary for the DeepSeek Qwen 14B model at temperature 0...")
    df_summary_deepseek_qwen_14b = df_summary[df_summary.index.get_level_values("target").str.startswith("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")]
    df_summary_deepseek_qwen_14b_temperature_0 = df_summary_deepseek_qwen_14b[df_summary_deepseek_qwen_14b.index.get_level_values("temperature") == 0]
    df_summary_deepseek_qwen_14b_temperature_0.to_csv("results_summary_deepseek_qwen_14b_temperature_0.csv", index=True)

    print(f"Stored all the results in {dirpath}.")
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)

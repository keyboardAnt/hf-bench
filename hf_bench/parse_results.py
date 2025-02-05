from pathlib import Path
import subprocess
from typing import Dict, List

import fire
import pandas as pd

from hf_bench.benchmark import ResultsTableRow


def get_columns() -> List[str]:
    return ResultsTableRow.__annotations__.keys()


def list_tracked_files(dirpath: str) -> Dict[str, str]:
    # Run git ls-tree command and capture output
    cmd = ["git", "ls-tree", "-r", "origin/main", "--name-only", dirpath]
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
    columns = ["submission_id"] + list(get_columns())
    df = pd.DataFrame(columns=columns)
    for f in filepaths:
        submission_id: str = Path(f).parent.stem
        df_new = pd.read_csv(f)
        df_new["submission_id"] = submission_id
        df_new = df_new[columns]
        df = pd.concat([df, df_new])
    df.sort_values(by=["target", "dataset_name", "drafter", "temperature", "example_id"], inplace=True)
    return df

def main(dirpath: str):
    df: pd.DataFrame = get_df_concat(dirpath)
    print(df)

if __name__ == "__main__":
    fire.Fire(main)

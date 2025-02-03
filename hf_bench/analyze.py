import fire
import pandas as pd
from scipy.stats import hmean


def is_harmonic_mean(column_name: str) -> bool:
    return column_name.endswith("TPOT")


def get_columns_for_ar(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if "AR " in col or "Baseline " in col]


def get_columns_for_ar_tpot(df: pd.DataFrame) -> list[str]:
    return [col for col in get_columns_for_ar(df) if "TPOT" in col]


def get_per_example_tpot_s_of_ar_star(df: pd.DataFrame) -> pd.DataFrame:
    """
    AR* denotes the AR setup of maximum TPOT for this example, among the two AR setups.
    In some CSVs, "AR" is written as "Baseline". The two AR setups differ by the temperature value, which some CSVs write as "`do_sample=bool`".
    """
    columns_for_ar = get_columns_for_ar_tpot(df=df)
    return df[columns_for_ar].max(axis=1)


def get_per_example_speedup_over_the_per_example_ar_star(df_per_example_tpot_s: pd.DataFrame, df_per_example_tpot_s_of_the_per_example_ar_star: pd.DataFrame) -> pd.DataFrame:
    """The average speedup is calculated as follows. For each input example and each algorithm, the speedup
    (i.e., "per-example speedup") is the ratio between the time-to-output-token (TPOT) of the AR* for this example (i.e., the AR setup with the maximum TPOT) and the TPOT of this algorithm."""
    return df_per_example_tpot_s.div(df_per_example_tpot_s_of_the_per_example_ar_star, axis=0)


def main(filepath_to_csv: str):
    df = pd.read_csv(filepath_to_csv)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 180)
    print("=" * 100)
    print("New Toks:")
    columns_for_new_toks = [
        col for col in df.columns if "New Toks" in col or "NewToks" in col
    ]
    print(df[columns_for_new_toks].mean().round(1))
    print("=" * 100)
    columns_for_ttft = [
        col for col in df.columns if "TTFT" in col
    ]
    print("TTFT (ms):")
    df_ttft_s = df[columns_for_ttft]
    print((df_ttft_s * 1000).mean().round(1))
    print("=" * 100)
    print("TPOT (ms), harmonic mean: (unsorted)")
    columns_for_tpot_s = [
        col for col in df.columns if "TPOT" in col
    ]
    df_tpot_s = df[columns_for_tpot_s]
    print((df_tpot_s * 1000).agg(hmean).round(1))
    print("=" * 100)
    print("Toks/sec: (unsorted)")
    df_tpot_s_hmean = df_tpot_s.agg(hmean)
    print((1/df_tpot_s_hmean).round(1))
    print("=" * 100)
    per_example_tpot_s_of_ar_star = get_per_example_tpot_s_of_ar_star(df=df)
    per_example_speedup_over_the_per_example_ar_star = get_per_example_speedup_over_the_per_example_ar_star(
        df_per_example_tpot_s=df_tpot_s, df_per_example_tpot_s_of_the_per_example_ar_star=per_example_tpot_s_of_ar_star)
    print("Per-example speedup over AR*: arithmetic mean (unsorted)")
    print(per_example_speedup_over_the_per_example_ar_star.mean().round(2))
    print("=" * 100)
    print("Per-example speedup over AR*: max (unsorted)")
    print(per_example_speedup_over_the_per_example_ar_star.max().round(2))


if __name__ == "__main__":
    fire.Fire(main)

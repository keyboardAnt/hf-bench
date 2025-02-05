from collections import defaultdict
import fire
import pandas as pd
from scipy.stats import hmean


def is_harmonic_mean(column_name: str) -> bool:
    return column_name.endswith("TPOT")


def get_columns_for_ar(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if "AR " in col or "Baseline " in col]


def get_columns_for_ar_tpot(df: pd.DataFrame) -> list[str]:
    return [col for col in get_columns_for_ar(df) if "TPOT" in col or "tpot_ms" in col]


def get_per_example_tpot_s_of_ar_star(df: pd.DataFrame) -> pd.DataFrame:
    """
    AR* denotes the AR setup of maximum TPOT for this example, among the two AR setups.
    In some CSVs, "AR" is written as "Baseline". The two AR setups differ by the temperature value, which some CSVs write as "`do_sample=bool`".
    """
    columns_for_ar = get_columns_for_ar_tpot(df=df)
    return df[columns_for_ar].max(axis=1)


def get_per_example_speedup_over_the_per_example_ar_star(
    df_per_example_tpot_s: pd.DataFrame,
    df_per_example_tpot_s_of_the_per_example_ar_star: pd.DataFrame,
) -> pd.DataFrame:
    """The average speedup is calculated as follows. For each input example and each algorithm, the speedup
    (i.e., "per-example speedup") is the ratio between the time-to-output-token (TPOT) of the AR* for this example (i.e., the AR setup with the maximum TPOT) and the TPOT of this algorithm."""
    return df_per_example_tpot_s.div(
        df_per_example_tpot_s_of_the_per_example_ar_star, axis=0
    )


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    assert len(df) > 0, f"No rows in {csv_path}"
    #num_examples = df["example_id"].nunique()
    tok_sec_ar = [None, None]
    df["drafter"] = df["drafter"].fillna("N/A (AR)").astype("category")
    draft2value = defaultdict(list)
    for drafter, df in dict(sorted(df.groupby("drafter", dropna=False, observed=False), key=lambda x: pd.isna(x[0]), reverse=True)).items():
        for temp, df in df.groupby("temperature"):
            id = f'{drafter=} {temp=}'
            print(f"\n\n{drafter=}, {temp=}\n")
            print("New Toks:")
            columns_for_new_toks = [
                col
                for col in df.columns
                if "New Toks" in col or "NewToks" in col or "new_toks" in col
            ]
            new_toks = df[columns_for_new_toks].mean().round(1)
            print(new_toks)
            draft2value[id].append(new_toks)
            print("=" * 100)
            columns_for_ttft = [
                col for col in df.columns if "TTFT" in col or "ttft_ms" in col
            ]
            print("TTFT (ms):")
            df_ttft_s = (df[columns_for_ttft] * 1000).mean().round(1)
            print(df_ttft_s)
            draft2value[id].append(df_ttft_s)
            print("=" * 100)
            print("TPOT (ms), harmonic mean:")
            columns_for_tpot_s = [
                col for col in df.columns if "TPOT" in col or "tpot_ms" in col
            ]
            df_tpot_s = (df[columns_for_tpot_s]* 1000).agg(hmean).round(1)
            print(df_tpot_s)
            draft2value[id].append(df_tpot_s)
            print("=" * 100)
            draft2value[id].append(-1)
            print("Toks/sec:")
            df_tpot_s = df[columns_for_tpot_s]
            df_tpot_s_hmean = df_tpot_s.agg(hmean)
            tok_sec = (1000 / df_tpot_s_hmean).round(1)
            print(tok_sec)
            draft2value[id].append(tok_sec)
            print("=" * 100)
            if drafter == "N/A (AR)":
                tok_sec_ar[temp] = tok_sec

            if tok_sec_ar is not None:
                print("Speedup:")
                speedup = (tok_sec / tok_sec_ar[temp]).round(2)
                print(speedup)
                draft2value[id].append(speedup)
                print("=" * 100)
            # Alternative for Toks/sec computation
            # columns_for_out_toks = [col for col in df.columns if "out_toks_per_sec" in col]
            # df_out_toks = df[columns_for_out_toks]
            # print(df_out_toks.agg(hmean).round(1))
            
            # No need for now
            # per_example_tpot_s_of_ar_star = get_per_example_tpot_s_of_ar_star(df=df)
            # per_example_speedup_over_the_per_example_ar_star = get_per_example_speedup_over_the_per_example_ar_star(
            #     df_per_example_tpot_s=df_tpot_s,
            #     df_per_example_tpot_s_of_the_per_example_ar_star=per_example_tpot_s_of_ar_star,
            # )
            # print("Per-example speedup over AR*: arithmetic mean (unsorted)")
            # print(per_example_speedup_over_the_per_example_ar_star.mean().round(2))
            # print("=" * 100)
            # print("Per-example speedup over AR*: max (unsorted)")
            # print(per_example_speedup_over_the_per_example_ar_star.max().round(2))
    print('Summary to copy in Google Sheets')
    for key, val in draft2value.items():
        float_list = [float(x.iloc[0]) if isinstance(x, pd.Series) else float(x) for x in val]
        print(f"{key}: {' '.join(map(str, float_list))}")

if __name__ == "__main__":
    fire.Fire(main)

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def compute_flags(df: pd.DataFrame, intrinsic_epsilon: float = 1e-4) -> pd.DataFrame:
    df = df.copy()
    df["invalid_spread"] = (df["bid"].notna() & df["ask"].notna() & (df["bid"] > df["ask"])) | (
        (df["bid"] < 0) | (df["ask"] < 0)
    )
    df["mid_missing"] = df["mid"].isna()
    df["intrinsic_violation"] = (df["mid"].notna()) & (df["intrinsic"].notna()) & (
        df["mid"] + intrinsic_epsilon < df["intrinsic"]
    )
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["quote_date", "expiration"])
    summary_rows: List[Dict] = []
    for keys, grp in grouped:
        quote_date, expiration = keys
        summary_rows.append(
            {
                "quote_date": quote_date,
                "expiration": expiration,
                "rows": len(grp),
                "invalid_spread": grp["invalid_spread"].sum(),
                "mid_missing": grp["mid_missing"].sum(),
                "intrinsic_violations": grp["intrinsic_violation"].sum(),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["quote_date", "expiration"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basic sanity checks on a normalized snapshot.")
    parser.add_argument("--input", required=True, help="Normalized Parquet input (single snapshot).")
    parser.add_argument("--output", required=True, help="CSV summary output.")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = compute_flags(df)
    summary = summarize(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print("Sanity summary (first 10 rows):")
    print(summary.head(10).to_string(index=False))
    print(f"\nWrote full summary to {output_path}")


if __name__ == "__main__":
    main()


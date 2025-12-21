import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # isort:skip


EPS = 1e-12


@dataclass
class DetectionResults:
    summary: pd.DataFrame
    samples: pd.DataFrame
    monotonicity: pd.DataFrame
    convexity: pd.DataFrame
    calendar: pd.DataFrame


def load_snapshot(input_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    if "option_type" not in df.columns:
        raise ValueError("input dataset missing option_type column")
    df = df[df["option_type"].isin(["call", "put"])].copy()
    df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.strftime("%Y-%m-%d")
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["quote_date", "expiration", "option_type", "strike"])
    return df


def _bucket_strike(series: pd.Series, width: float) -> pd.Series:
    width = width if width > 0 else 1.0
    return np.round(series / width) * width


def _monotonicity_records(df: pd.DataFrame, quote_date: str, expiration: str, option_type: str) -> List[Dict]:
    g = df.sort_values("strike")
    strikes = g["strike"].to_numpy()
    bids = g["bid"].to_numpy()
    asks = g["ask"].to_numpy()
    records: List[Dict] = []
    if len(g) < 2:
        return records

    for i in range(len(g) - 1):
        k_curr, k_next = strikes[i], strikes[i + 1]
        bid_curr, bid_next = bids[i], bids[i + 1]
        ask_curr, ask_next = asks[i], asks[i + 1]

        if option_type == "call":
            if np.isnan(bid_next) or np.isnan(ask_curr):
                continue
            if bid_next > ask_curr + EPS:
                records.append(
                    {
                        "quote_date": quote_date,
                        "expiration": expiration,
                        "option_type": option_type,
                        "strike": k_next,
                        "strike_low": k_curr,
                        "strike_high": k_next,
                        "bid_high": bid_next,
                        "ask_low": ask_curr,
                        "detail": "high-strike bid exceeds low-strike ask",
                    }
                )
        else:  # put prices should increase with strike
            if np.isnan(bid_curr) or np.isnan(ask_next):
                continue
            if bid_curr > ask_next + EPS:
                records.append(
                    {
                        "quote_date": quote_date,
                        "expiration": expiration,
                        "option_type": option_type,
                        "strike": k_curr,
                        "strike_low": k_curr,
                        "strike_high": k_next,
                        "bid_low": bid_curr,
                        "ask_high": ask_next,
                        "detail": "low-strike bid exceeds high-strike ask",
                    }
                )
    return records


def _convexity_records(df: pd.DataFrame, quote_date: str, expiration: str, option_type: str) -> List[Dict]:
    g = df.sort_values("strike")
    strikes = g["strike"].to_numpy()
    bids = g["bid"].to_numpy()
    asks = g["ask"].to_numpy()
    records: List[Dict] = []
    if len(g) < 3:
        return records

    for i in range(len(g) - 2):
        k1, k2, k3 = strikes[i], strikes[i + 1], strikes[i + 2]
        b1, a2, b3 = bids[i], asks[i + 1], bids[i + 2]
        if any(np.isnan(v) for v in (b1, a2, b3)):
            continue
        if (k2 - k1) == 0 or (k3 - k2) == 0:
            continue
        slope1 = (a2 - b1) / (k2 - k1)
        slope2 = (b3 - a2) / (k3 - k2)
        if slope1 + EPS < slope2:
            records.append(
                {
                    "quote_date": quote_date,
                    "expiration": expiration,
                    "option_type": option_type,
                    "strike": k2,
                    "k1": k1,
                    "k2": k2,
                    "k3": k3,
                    "bid_k1": b1,
                    "ask_k2": a2,
                    "bid_k3": b3,
                    "slope_left": slope1,
                    "slope_right": slope2,
                    "detail": "discrete convexity violated even with best-case bounds",
                }
            )
    return records


def _calendar_records(df: pd.DataFrame) -> List[Dict]:
    records: List[Dict] = []
    grouped = df.groupby(["quote_date", "option_type", "strike"])
    for (quote_date, option_type, strike), grp in grouped:
        g = grp.sort_values("expiration")
        expirations = g["expiration"].to_numpy()
        bids = g["bid"].to_numpy()
        asks = g["ask"].to_numpy()
        for i in range(len(g) - 1):
            short_exp, long_exp = expirations[i], expirations[i + 1]
            bid_short, ask_long = bids[i], asks[i + 1]
            if np.isnan(bid_short) or np.isnan(ask_long):
                continue
            if ask_long + EPS < bid_short:
                records.append(
                    {
                        "quote_date": quote_date,
                        "expiration": short_exp,
                        "expiration_long": long_exp,
                        "option_type": option_type,
                        "strike": strike,
                        "bid_short": bid_short,
                        "ask_long": ask_long,
                        "detail": "longer-dated ask below shorter-dated bid",
                    }
                )
    return records


def summarize(base_df: pd.DataFrame, mono_df: pd.DataFrame, conv_df: pd.DataFrame, cal_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        base_df.groupby(["quote_date", "expiration", "option_type"])
        .size()
        .reset_index(name="rows")
    )

    def add_count(src: pd.DataFrame, col_name: str) -> None:
        nonlocal summary
        if src.empty:
            summary[col_name] = 0
            return
        counts = (
            src.groupby(["quote_date", "expiration", "option_type"])
            .size()
            .reset_index(name=col_name)
        )
        summary = summary.merge(counts, on=["quote_date", "expiration", "option_type"], how="left")
        summary[col_name] = summary[col_name].fillna(0).astype(int)

    add_count(mono_df, "monotonicity_violations")
    add_count(conv_df, "convexity_violations")
    if cal_df.empty:
        summary["calendar_violations"] = 0
    else:
        cal_counts = (
            cal_df.groupby(["quote_date", "expiration", "option_type"])
            .size()
            .reset_index(name="calendar_violations")
        )
        summary = summary.merge(cal_counts, on=["quote_date", "expiration", "option_type"], how="left")
        summary["calendar_violations"] = summary["calendar_violations"].fillna(0).astype(int)

    return summary.sort_values(["quote_date", "expiration", "option_type"])


def _plot_heatmap(counts: pd.DataFrame, output_path: Path, title: str) -> None:
    if counts.empty:
        return
    pivot = counts.pivot(index="expiration", columns="strike_bucket", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 0.35), max(4, pivot.shape[0] * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(s) for s in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Strike bucket")
    ax.set_ylabel("Expiration")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="violations")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _heatmap_counts(df: pd.DataFrame, bucket_width: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["strike_bucket"] = _bucket_strike(out["strike"], bucket_width)
    out["count"] = 1
    return (
        out.groupby(["expiration", "strike_bucket"])["count"]
        .sum()
        .reset_index()
        .sort_values(["expiration", "strike_bucket"])
    )


def run_detection(
    input_path: Path,
    output_dir: Path,
    sample_size: int = 200,
    strike_bucket: float = 1.0,
) -> DetectionResults:
    df = load_snapshot(input_path)

    mono_records: List[Dict] = []
    conv_records: List[Dict] = []
    for (quote_date, expiration, option_type), grp in df.groupby(["quote_date", "expiration", "option_type"]):
        mono_records.extend(_monotonicity_records(grp, quote_date, expiration, option_type))
        conv_records.extend(_convexity_records(grp, quote_date, expiration, option_type))

    cal_records = _calendar_records(df)

    mono_df = pd.DataFrame(mono_records)
    conv_df = pd.DataFrame(conv_records)
    cal_df = pd.DataFrame(cal_records)

    summary = summarize(df, mono_df, conv_df, cal_df)

    samples = pd.concat(
        [
            mono_df.assign(rule="monotonicity"),
            conv_df.assign(rule="convexity"),
            cal_df.assign(rule="calendar"),
        ],
        ignore_index=True,
    )
    if not samples.empty:
        samples = samples.sort_values(["quote_date", "expiration", "option_type"]).head(sample_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    samples.to_csv(output_dir / "sample_violations.csv", index=False)

    _plot_heatmap(_heatmap_counts(mono_df, strike_bucket), output_dir / "monotonicity_heatmap.png", "Monotonicity violations")
    _plot_heatmap(_heatmap_counts(conv_df, strike_bucket), output_dir / "convexity_heatmap.png", "Convexity violations")
    _plot_heatmap(_heatmap_counts(cal_df, strike_bucket), output_dir / "calendar_heatmap.png", "Calendar violations")

    return DetectionResults(
        summary=summary,
        samples=samples,
        monotonicity=mono_df,
        convexity=conv_df,
        calendar=cal_df,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Detect static arbitrage violations with bid-ask awareness.")
    parser.add_argument("--input", required=True, help="Parquet snapshot input path.")
    parser.add_argument("--output-dir", required=True, help="Directory for reports and plots.")
    parser.add_argument("--sample-size", type=int, default=200, help="Max sample rows to export.")
    parser.add_argument("--strike-bucket", type=float, default=1.0, help="Strike bucket size for heatmaps.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    results = run_detection(input_path, output_dir, sample_size=args.sample_size, strike_bucket=args.strike_bucket)

    print("Summary (first 10 rows):")
    print(results.summary.head(10).to_string(index=False))
    print(f"\nWrote summary to {output_dir / 'summary.csv'}")
    print(f"Wrote sample violations to {output_dir / 'sample_violations.csv'}")
    print(f"Wrote heatmaps under {output_dir}")


if __name__ == "__main__":
    main()


from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go


def _load_and_pivot(grid_csv: Path, params_csv: Path, quote_date: Optional[str] = None):
    grid_df = pd.read_csv(grid_csv)
    params_df = pd.read_csv(params_csv)

    if quote_date:
        grid_df = grid_df[grid_df["quote_date"] == quote_date]
        params_df = params_df[params_df["quote_date"] == quote_date]
        if grid_df.empty:
            raise ValueError(f"No data for quote_date={quote_date}")

    # map expiration -> ttm_years from params
    ttm_map = dict(zip(params_df["expiration"], params_df["ttm_years"]))
    grid_df["ttm_years"] = grid_df["expiration"].map(ttm_map)

    # pivot to strike (X) x ttm (Y) mesh with iv_fit as Z
    pivot = grid_df.pivot_table(index="ttm_years", columns="strike", values="iv_fit")
    # ensure sorted axes
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    Y = np.array(pivot.index)
    X = np.array(pivot.columns)
    Z = pivot.values
    return X, Y, Z


def plot_iv_surface(
    grid_csv: str | Path,
    params_csv: str | Path,
    quote_date: Optional[str] = None,
    title: Optional[str] = None,
    output_html: Optional[str | Path] = None,
) -> go.Figure:
    """
    Render a Plotly 3D IV surface from grid_fit + fit_params outputs.

    Args:
        grid_csv: path to grid_fit.csv (fitted IVs on dense grid)
        params_csv: path to fit_params.csv (includes ttm_years per expiration)
        quote_date: optional filter for a specific quote_date
        title: optional plot title
        output_html: if provided, write an interactive HTML file
    """
    X, Y, Z = _load_and_pivot(Path(grid_csv), Path(params_csv), quote_date)
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
    fig.update_layout(
        title=title or f"IV Surface{f' {quote_date}' if quote_date else ''}",
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="TTM (years)",
            zaxis_title="Implied Vol",
        ),
        autosize=True,
    )
    if output_html:
        fig.write_html(str(output_html))
    return fig


"""Data quality assessment and preprocessing using Polars and DuckDB."""

import duckdb
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


def assess_data_quality(df: pl.DataFrame, value_col: str) -> Dict:
    """Quality metrics via DuckDB SQL; duplicate count via Polars."""
    stats = duckdb.sql(f"""
        WITH bounds AS (
            SELECT
                QUANTILE_CONT("{value_col}", 0.01) AS q01,
                QUANTILE_CONT("{value_col}", 0.99) AS q99
            FROM df
        )
        SELECT
            COUNT(*) - COUNT(d."{value_col}")                                           AS missing_values,
            100.0 * (COUNT(*) - COUNT(d."{value_col}")) / COUNT(*)                     AS missing_pct,
            SUM(CASE WHEN d."{value_col}" > b.q99
                      OR d."{value_col}" < b.q01 THEN 1 ELSE 0 END)                    AS outliers,
            MAX(d."{value_col}") - MIN(d."{value_col}")                                 AS data_range,
            VAR_SAMP(d."{value_col}")                                                   AS variance
        FROM df d, bounds b
    """).pl().row(0, named=True)

    return {
        **stats,
        "duplicates": df.height - df.unique().height,
    }


def preprocess_time_series(df: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """Deduplicate, forward/back fill nulls, clip IQR outliers — bounds from DuckDB."""
    bounds = duckdb.sql(f"""
        SELECT
            QUANTILE_CONT("{value_col}", 0.25) - 1.5
                * (QUANTILE_CONT("{value_col}", 0.75) - QUANTILE_CONT("{value_col}", 0.25)) AS lower_bound,
            QUANTILE_CONT("{value_col}", 0.75) + 1.5
                * (QUANTILE_CONT("{value_col}", 0.75) - QUANTILE_CONT("{value_col}", 0.25)) AS upper_bound
        FROM df
    """).pl().row(0, named=True)

    return (
        df.unique()
          .with_columns(
              pl.col(value_col)
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
          )
          .with_columns(
              pl.col(value_col).clip(bounds["lower_bound"], bounds["upper_bound"])
          )
    )


def plot_data_quality(
    original: pl.Series,
    processed: pl.Series,
    title: str,
    output_path: Path,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(original.to_list(), color="#4A90A4", linewidth=1.2, alpha=0.7)
    ax1.set_ylabel("Value")
    ax1.set_title("Original")

    ax2.plot(processed.to_list(), color="#D4A574", linewidth=1.2, alpha=0.7)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    ax2.set_title("After Preprocessing")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()

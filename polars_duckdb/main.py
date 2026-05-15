#!/usr/bin/env python3
"""Data quality assessment and preprocessing — Polars + DuckDB rewrite."""

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from core import assess_data_quality, plot_data_quality, preprocess_time_series

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Data quality — Polars + DuckDB")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    value_col = config["data"]["value_column"]
    date_col = config["data"]["date_column"]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["output"]["figures_dir"])
    )
    output_dir.mkdir(exist_ok=True)

    if args.data_path and args.data_path.exists():
        df = pl.read_csv(args.data_path, try_parse_dates=True)
    elif config["data"]["generate_synthetic"]:
        rng = np.random.default_rng(config["data"]["seed"])
        n = config["data"]["n_periods"]
        start = date(2023, 1, 1)
        dates = [start + timedelta(days=i) for i in range(n)]
        values = np.sin(np.arange(n) / 10) + rng.normal(0, 0.2, n)
        # inject missing values and spikes
        null_idx = rng.choice(n, 10, replace=False).tolist()
        spike_idx = rng.choice(n, 5, replace=False).tolist()
        values[spike_idx] += 10
        values_list = [
            None if i in null_idx else float(v) for i, v in enumerate(values)
        ]
        df = pl.DataFrame({date_col: dates, value_col: values_list})
    else:
        raise ValueError("No data source specified")

    quality = assess_data_quality(df, value_col)
    logging.info("Data Quality Metrics:")
    logging.info(
        f"  Missing values : {quality['missing_values']} ({quality['missing_pct']:.2f}%)"
    )
    logging.info(f"  Duplicates     : {quality['duplicates']}")
    logging.info(f"  Outliers       : {quality['outliers']}")
    logging.info(f"  Data range     : {quality['data_range']:.4f}")
    logging.info(f"  Variance       : {quality['variance']:.4f}")

    df_processed = preprocess_time_series(df, value_col)

    quality_after = assess_data_quality(df_processed, value_col)
    logging.info("After Preprocessing:")
    logging.info(f"  Missing values : {quality_after['missing_values']}")
    logging.info(f"  Outliers       : {quality_after['outliers']}")

    plot_data_quality(
        df[value_col],
        df_processed[value_col],
        "Data Quality: Before and After Preprocessing",
        output_dir / "data_quality_comparison.png",
    )

    logging.info(f"Analysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

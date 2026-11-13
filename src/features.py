"""Time-series feature engineering utilities for lithium price forecasting."""
import pandas as pd

def create_lagged_column(series: pd.Series, lag: int, col_name: str) -> pd.Series:
    """Return one lagged copy of a series."""
    if lag < 1:
        raise ValueError("lag must be >= 1 to avoid using contemporaneous/future values")
    return series.shift(lag).rename(f"{col_name}_{lag}")

def create_lagged_features(
    df: pd.DataFrame,
    columns=None,
    min_lag: int = 1,
    max_lag: int = 12,
    drop_all_missing_rows: bool = False,
) -> pd.DataFrame:
    """
    Create lagged columns for selected input series.

    Missing values are preserved by default. Do not globally drop rows just
    because an optional predictor is missing; fit imputers on training data
    inside the model pipeline instead.
    """
    if min_lag < 1 or max_lag < min_lag:
        raise ValueError("Require 1 <= min_lag <= max_lag")
    if columns is None:
        columns = df.columns.tolist()

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    lagged_frames = [
        create_lagged_column(df[col], lag, col)
        for col in columns
        for lag in range(min_lag, max_lag + 1)
    ]
    if not lagged_frames:
        return pd.DataFrame(index=df.index)
    lagged_df = pd.concat(lagged_frames, axis=1)
    if drop_all_missing_rows:
        # Only removes rows where every generated feature is missing.
        lagged_df = lagged_df.dropna(how="all")
    return lagged_df

import pandas as pd
import numpy as np
import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate columns from a DataFrame (usually after merges).
    """
    return df.loc[:, ~df.columns.duplicated()].copy()

# Default categorical dimensions to use for the estimation grid
CAT_DIMS_DEFAULT = ["objective", "publisher_platform", "age", "gender"]

def is_wildcard_row(row: pd.Series, dims: Sequence[str]) -> bool:
    """
    Check if a row contains any NA values in the categorical dimensions.
    These rows are treated as wildcard rows (aggregated across dimensions).
    """
    return any(pd.isna(row[d]) for d in dims)

def _specific_mask(df: pd.DataFrame, dims: Sequence[str]) -> pd.Series:
    """
    Return a boolean mask identifying rows with no NA in any categorical dimension.
    """
    return df[dims].notna().all(axis=1)

def _match_partial(row: pd.Series, dims: Sequence[str], candidate: pd.DataFrame) -> pd.Series:
    """
    Return a mask over 'candidate' that matches all non-NA values in 'row'.
    NA acts as a wildcard, accepting any value.
    """
    mask = pd.Series(True, index=candidate.index)
    for d in dims:
        if pd.isna(row[d]):
            continue
        mask &= (candidate[d] == row[d])
    return mask

def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize a vector of weights to sum to 1.
    Fall back to uniform distribution if weights are invalid.
    """
    total = float(np.nansum(weights))
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(weights) / len(weights)
    return weights / total

def estimate_values_sum_preserving(
    df_original: pd.DataFrame,
    cartesian: pd.DataFrame,
    dims: Sequence[str] = CAT_DIMS_DEFAULT,
    metrics: Sequence[str] = ("impressions", "clicks", "spend"),
    ad_col: str = "ad_id",
) -> pd.DataFrame:
    """
    Estimate missing rows per ad_id by projecting original rows onto a complete
    categorical grid (Cartesian product) while preserving total metric sums.
    
    Parameters
    ----------
    df_original : DataFrame
        The original input data, possibly with missing categorical combinations.
    cartesian : DataFrame
        The full expected grid of combinations per ad_id.
    dims : list[str]
        Categorical dimensions used to define the space.
    metrics : list[str]
        Metrics to estimate (e.g., impressions, clicks, spend).
    ad_col : str
        The primary ID column used to separate ads.

    Returns
    -------
    DataFrame
        A completed version of the dataset, with all combinations filled and
        metric totals preserved per ad.
    """
    base_cols = [ad_col] + list(dims)
    keep_cols = list(dict.fromkeys(base_cols + list(metrics)))
    output = []

    for ad, group in df_original.groupby(ad_col, dropna=False):
        try:
            grid = cartesian[cartesian[ad_col] == ad].copy()
            specific_rows = group[_specific_mask(group, dims)]
            filled = grid[base_cols].copy()

            # Initialize all metric columns with zeros
            for m in metrics:
                filled[m] = 0.0

            # Step 1: Retain fully-specified rows (no missing dims)
            if not specific_rows.empty:
                merged = filled.merge(specific_rows[keep_cols], on=base_cols, how="left", suffixes=("", "_obs"))
                for m in metrics:
                    merged[m] = merged[f"{m}_obs"].fillna(0.0)
                    merged.drop(columns=[f"{m}_obs"], inplace=True)
                filled = merged

            # Step 2: Expand wildcard rows (with NA in dims)
            wildcard_rows = group.loc[~_specific_mask(group, dims), keep_cols].copy()
            candidates = filled[base_cols].copy()

            for _, wc in wildcard_rows.iterrows():
                match_mask = _match_partial(wc, dims, candidates)
                target_idx = candidates.index[match_mask]
                if len(target_idx) == 0:
                    logger.warning(f"No matching candidates for wildcard row (ad_id={ad}). Skipping.")
                    continue
                for m in metrics:
                    existing = filled.loc[target_idx, m].to_numpy()
                    weights = _normalize_weights(existing)
                    val = float(wc[m]) if pd.notna(wc[m]) else 0.0
                    filled.loc[target_idx, m] += weights * val

            # Step 3: Fix any residual delta to exactly preserve total
            for m in metrics:
                original_sum = group[m].fillna(0.0).sum()
                estimated_sum = filled[m].fillna(0.0).sum()
                delta = original_sum - estimated_sum
                if abs(delta) > 1e-9 and len(filled) > 0:
                    max_idx = filled[m].abs().idxmax()
                    filled.loc[max_idx, m] += delta

            output.append(filled[keep_cols])

        except Exception as e:
            logger.exception(f"Error while estimating values for ad_id={ad}: {e}")
            raise

    return pd.concat(output, ignore_index=True)

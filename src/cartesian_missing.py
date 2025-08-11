# step2_cartesian.py
"""
Step 2: Cartesian Product & Missing Combination Detection

This module:
1. Normalizes categorical and date columns for consistency.
2. Builds the full Cartesian product of categorical dimensions + date ranges per ad_id.
3. Detects missing combinations in event tables (with NA treated as a wildcard).

Author: Your Name
"""

import logging
import pandas as pd
import numpy as np
from itertools import product
from typing import List, Optional

logger = logging.getLogger(__name__)

SENTINEL = "__NA__"

__all__ = [
    "normalize_keys",
    "build_cartesian_dates_as_pairs",
    "find_missing_flexible_per_ad",
]


def normalize_keys(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Normalize categorical and date columns for consistent comparison.

    - Converts 'objective', 'age', 'gender' to pandas StringDtype and replaces empty/None/nan with NA.
    - Parses 'date_start'/'date_stop' to datetime, formats as 'YYYY-MM-DD', and replaces NaT with NA.

    Parameters
    ----------
    df : pd.DataFrame or None
        Input dataframe to normalize.

    Returns
    -------
    pd.DataFrame or None
        Normalized dataframe or None if input is None/empty.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    try:
        # Normalize categorical dims
        for c in ["objective", "age", "gender"]:
            if c in out.columns:
                out[c] = out[c].astype("string")
                out.loc[out[c].isin(["", "None", "nan"]), c] = pd.NA

        # Normalize dates
        for c in ["date_start", "date_stop"]:
            if c in out.columns:
                dt = pd.to_datetime(out[c], errors="coerce")
                out[c] = dt.dt.strftime("%Y-%m-%d").astype("string")
                out.loc[dt.isna(), c] = pd.NA

    except Exception as e:
        logger.exception(f"Error in normalize_keys: {e}")
        raise
    return out


def _collect_period_pairs_for_ad(ad_id, sources: List[pd.DataFrame]):
    """
    Collect all unique (date_start, date_stop) pairs for a given ad_id.
    """
    pairs = set()
    for df in sources:
        if df is None or df.empty:
            continue
        sub = df[df["ad_id"] == ad_id]
        if {"date_start", "date_stop"}.issubset(sub.columns):
            for ds, de in zip(sub["date_start"], sub["date_stop"]):
                ds = None if pd.isna(ds) else ds
                de = None if pd.isna(de) else de
                pairs.add((ds, de))
    return list(pairs)


def _values_by_dim_for_ad(ad_id, dims: List[str], sources: List[pd.DataFrame]):
    """
    Collect unique values for each categorical dimension for a given ad_id.
    """
    by_dim = {d: set() for d in dims}
    for df in sources:
        if df is None or df.empty:
            continue
        sub = df[df["ad_id"] == ad_id]
        for d in dims:
            if d in sub.columns:
                by_dim[d].update(sub[d].dropna().unique().tolist())
    return {d: (sorted(list(v)) if v else []) for d, v in by_dim.items()}


def build_cartesian_dates_as_pairs(
    df_core: pd.DataFrame,
    event_tables: Optional[List[pd.DataFrame]] = None,
    dims: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build full cartesian product of dims for each ad_id + all real date pairs.

    Parameters
    ----------
    df_core : pd.DataFrame
        Core dataframe with at least 'ad_id' and the dims/date columns.
    event_tables : list of pd.DataFrame, optional
        Additional event tables to consider when extracting values.
    dims : list of str, optional
        Dimensions to include in the Cartesian product.

    Returns
    -------
    pd.DataFrame
        Cartesian table with all combinations for each ad_id and date pair.
    """
    if dims is None:
        dims = ["objective", "age", "gender"]

    all_sources = [df_core] + ([t for t in event_tables if t is not None] if event_tables else [])
    rows = []

    try:
        for ad_id in df_core["ad_id"].dropna().unique():
            period_pairs = _collect_period_pairs_for_ad(ad_id, all_sources)
            if not period_pairs:
                period_pairs = [(pd.NA, pd.NA)]
            space = _values_by_dim_for_ad(ad_id, dims, all_sources)
            lists = {d: (space[d] if space[d] else [pd.NA]) for d in dims}

            for ds, de in period_pairs:
                for combo in product(*(lists[d] for d in dims)):
                    rec = {"ad_id": ad_id, "date_start": ds, "date_stop": de}
                    rec.update({d: v for d, v in zip(dims, combo)})
                    rows.append(rec)

    except Exception as e:
        logger.exception(f"Error building cartesian: {e}")
        raise

    df_cart = pd.DataFrame(rows, columns=["ad_id", "date_start", "date_stop"] + dims)
    logger.info(f"Cartesian built with {len(df_cart)} rows")
    return df_cart


def find_missing_flexible_per_ad(cart: pd.DataFrame, df_event: pd.DataFrame, dims: List[str]) -> pd.DataFrame:
    """
    Identify missing combinations from cart vs event table (NA in event table = wildcard).

    Parameters
    ----------
    cart : pd.DataFrame
        Full cartesian table.
    df_event : pd.DataFrame
        Event table with actual combinations.
    dims : list of str
        Dimensions to compare (should include date_start/date_stop if needed).

    Returns
    -------
    pd.DataFrame
        Rows from cart that are not covered by the event table.
    """
    keys = ["ad_id"] + dims
    missing_parts = []

    try:
        cart2 = cart.loc[:, ~cart.columns.duplicated()]
        ev = df_event.loc[:, ~df_event.columns.duplicated()]

        # Ensure all key columns exist
        for k in keys:
            if k not in cart2.columns:
                cart2[k] = pd.NA
            if k not in ev.columns:
                ev[k] = pd.NA

        # Work with unique rows only
        cart2 = cart2[keys].drop_duplicates().copy()
        ev2 = ev[keys].drop_duplicates().copy()

        for ad_id, cart_g in cart2.groupby("ad_id", dropna=False):
            ev_g = ev2[ev2["ad_id"] == ad_id]
            if ev_g.empty:
                missing_parts.append(cart_g)
                continue

            covered = np.zeros(len(cart_g), dtype=bool)
            cart_cols = {d: cart_g[d].values for d in dims}

            for _, rule in ev_g.iterrows():
                m = np.ones(len(cart_g), dtype=bool)
                for d in dims:
                    val = rule[d]
                    if isinstance(val, (pd.Series, np.ndarray, list, tuple)):
                        allowed = pd.Series(val).dropna().unique()
                    else:
                        allowed = [] if pd.isna(val) else [val]
                    if not allowed:
                        continue
                    m &= np.isin(cart_cols[d], allowed)
                    if not m.any():
                        break
                covered |= m
                if covered.all():
                    break

            missing_parts.append(cart_g[~covered])

    except Exception as e:
        logger.exception(f"Error finding missing combos: {e}")
        raise

    result = pd.concat(missing_parts, ignore_index=True) if missing_parts else cart.iloc[0:0]
    logger.info(f"Missing combinations found: {len(result)}")
    return result

def build_cart(df_core: pd.DataFrame, df_events: Optional[pd.DataFrame] = None):
    return build_cartesian_dates_as_pairs(df_core, event_tables=[df_events] if df_events is not None else None)

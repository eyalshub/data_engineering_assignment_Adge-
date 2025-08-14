# src/cartesian_missing.py
from __future__ import annotations
import itertools
from typing import Dict, List, Iterable, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Categorical dimensions ONLY (as required by the assignment)
DIMS = ["objective", "publisher_platform", "age", "gender"]

PLATFORM_ALIASES = {
    "audience network": "audience_network",
    "audience_network": "audience_network",
    "fb": "facebook",
    # אם תרצה: "threads": "instagram",
}
PLATFORM_ALLOW = {"facebook", "instagram", "audience_network", "messenger", "threads"}

def _clean_value(dim: str, v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip()
    if s == "":
        return pd.NA
    s = s.lower()
    if s in {"none", "nan", "unknown", "unspecified"}:
        return pd.NA
    if dim == "publisher_platform":
        s = PLATFORM_ALIASES.get(s, s)
        if s not in PLATFORM_ALLOW:
            return pd.NA
    return s


__all__ = [
    "DIMS",
    "domains_per_ad",
    "build_cartesian",
    "find_missing",
]

# -----------------------
# Helpers / Normalization
# -----------------------
def _normalize_dims(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in DIMS:
        if c in out.columns:
            out[c] = out[c].map(lambda x: _clean_value(c, x)).astype("string")
    return out

def _unique_non_null(vals: Iterable) -> List:
    """Unique, non-null values preserving stable order."""
    s = pd.Series(list(vals))
    s = s.dropna()
    return list(pd.unique(s))


# -----------------------
# Public API
# -----------------------
def domains_per_ad(
    df_core: pd.DataFrame,
    extra_sources: List[pd.DataFrame] | None = None,
    dims: List[str] = DIMS,
) -> Dict[int, Dict[str, List]]:
    """
    Build per-ad domains for categorical dims:
    domain(ad, dim) = unique values observed for that ad across all provided sources.
    If no value observed for a given dim, the domain falls back to [NA] (wildcard-only).
    """
    sources = [df_core] + (extra_sources or [])
    sources = [_normalize_dims(s) for s in sources if s is not None]

    # Collect all ad_ids across sources
    ad_ids: set = set()
    for s in sources:
        if "ad_id" in s.columns:
            ad_ids |= set(s["ad_id"].dropna().unique().tolist())

    out: Dict[int, Dict[str, List]] = {}
    for ad in ad_ids:
        out[ad] = {}
        for d in dims:
            vals: List = []
            for s in sources:
                if {"ad_id", d}.issubset(s.columns):
                    vals.extend(s.loc[s["ad_id"] == ad, d].dropna().tolist())
            uniq = _unique_non_null(vals)
            out[ad][d] = uniq if len(uniq) > 0 else [pd.NA]
    return out


def build_cartesian(
    df_core: pd.DataFrame,
    extra_sources: List[pd.DataFrame] | None = None,
    dims: List[str] = DIMS,
) -> pd.DataFrame:
    """
    Build the full Cartesian product per ad_id over the categorical dims only.
    No dates, no numeric metrics here.
    """
    df_core = _normalize_dims(df_core)
    per = domains_per_ad(df_core, extra_sources, dims)

    frames: List[pd.DataFrame] = []
    for ad, doms in per.items():
        prod = list(itertools.product(*[doms[d] for d in dims]))
        df = pd.DataFrame(prod, columns=dims)
        df.insert(0, "ad_id", ad)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ad_id"] + dims)
    logger.info(f"Cartesian grid built: shape={out.shape}")
    return out


def _covered_combos_for_row(
    row: pd.Series,
    ad_domain: Dict[str, List],
    dims: List[str],
) -> Iterable[Tuple]:
    """
    Expand a single existing row into all combos it *covers* under NA-as-wildcard semantics:
    - If row[dim] is NA -> covers the entire domain for that dim (for this ad).
    - Else -> covers exactly the observed value on that dim.
    """
    choices: List[List] = []
    for d in dims:
        v = row[d]
        if pd.isna(v):
            choices.append(ad_domain[d])  # wildcard expands over the ad-specific domain
        else:
            choices.append([v])
    for combo in itertools.product(*choices):
        yield combo


def find_missing(
    df_core: pd.DataFrame,
    cartesian: pd.DataFrame,
    dims: List[str] = DIMS,
    extra_sources: List[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Identify missing combinations per ad_id:
    Missing = (full cartesian) - (combinations covered by existing rows),
    where NA in existing data acts as a wildcard (covers whole domain for that dim).
    """
    df_core = _normalize_dims(df_core)
    cartesian = _normalize_dims(cartesian)
    per = domains_per_ad(df_core, extra_sources, dims)

    missing_parts: List[pd.DataFrame] = []

    for ad, cart_g in cartesian.groupby("ad_id", dropna=False):
        all_set = set(map(tuple, cart_g[dims].itertuples(index=False, name=None)))

        existing = df_core[df_core["ad_id"] == ad]
        covered: set = set()
        for _, r in existing.iterrows():
            for combo in _covered_combos_for_row(r, per[ad], dims):
                covered.add(tuple(combo))

        miss = all_set - covered
        if miss:
            mdf = pd.DataFrame(list(miss), columns=dims)
            mdf.insert(0, "ad_id", ad)
            missing_parts.append(mdf)

    result = pd.concat(missing_parts, ignore_index=True) if missing_parts else pd.DataFrame(columns=["ad_id"] + dims)
    logger.info(f"Missing combos found: shape={result.shape}")
    return result



# --- Convenience helpers (per-ad) ---

def build_cartesian_for_ad_from_domain(ad_id: int, ad_domain: Dict[str, List], dims: List[str] = DIMS) -> pd.DataFrame:
    """
    Build the cartesian grid for a single ad_id given its per-ad domain dict.
    """
    prod = list(itertools.product(*[ad_domain[d] for d in dims]))
    df = pd.DataFrame(prod, columns=dims)
    df.insert(0, "ad_id", ad_id)
    return df


def build_cartesian_for_ad(df_core: pd.DataFrame, ad_id: int,
                           extra_sources: List[pd.DataFrame] | None = None,
                           dims: List[str] = DIMS) -> pd.DataFrame:
    """
    Compute per-ad domain (optionally across extra sources) and build the grid for that single ad_id.
    """
    per = domains_per_ad(df_core, extra_sources=extra_sources, dims=dims)
    if ad_id not in per:
        return pd.DataFrame(columns=["ad_id"] + dims)
    return build_cartesian_for_ad_from_domain(ad_id, per[ad_id], dims=dims)


def find_missing_for_ad(df_core: pd.DataFrame, ad_id: int,
                        extra_sources: List[pd.DataFrame] | None = None,
                        dims: List[str] = DIMS) -> pd.DataFrame:
    """
    Compute missing combos for a single ad_id. Uses NA-as-wildcard semantics from find_missing.
    """
    # Build a cartesian only for this ad
    cart = build_cartesian_for_ad(df_core, ad_id, extra_sources=extra_sources, dims=dims)
    if cart.empty:
        return cart
    # Filter df_core to this ad and reuse find_missing logic
    df_core_ad = df_core[df_core["ad_id"] == ad_id].copy()
    return find_missing(df_core_ad, cart, dims=dims, extra_sources=extra_sources)

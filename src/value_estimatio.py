# Step 3: Value estimation (with optional Neighbors fallback)
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "drop_duplicate_columns",
    "_safe_norm",
    "build_neighbors_index",
    "_weights_age_gender",
    "estimate_core_metrics",
    "estimate_event_table",
]

# -----------------------------
# Utilities
# -----------------------------

def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate columns from a DataFrame."""
    return df.loc[:, ~df.columns.duplicated()].copy()

def _safe_norm(w):
    """Normalize weight vector, fallback to uniform if sum <= 0 or invalid."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        logger.debug("Weights sum is zero or invalid, using uniform distribution.")
        return np.ones_like(w) / len(w)
    return w / s

# -----------------------------
# Neighbors index (optional fallback)
# -----------------------------

def _zscore_fit(vals: pd.Series):
    m = np.nanmean(vals)
    s = np.nanstd(vals)
    if not np.isfinite(s) or s == 0:
        s = 1.0
    return float(m), float(s)

def _zscore_apply(x, m, s):
    return (x - m) / s

def _safe_prop(s: pd.Series):
    total = s.sum()
    if total <= 0 or not np.isfinite(total):
        return pd.Series(np.ones(len(s)) / max(len(s), 1), index=s.index)
    return s / total

def build_neighbors_index(core_ag: pd.DataFrame,
                          feature_cols=("impressions", "clicks", "spend"),
                          dims=("objective",)):
    """
    Build a lightweight neighbors index for fallback weighting.

    Parameters
    ----------
    core_ag : DataFrame
        Rows with age & gender populated (granular level, not platform rows).
    feature_cols : tuple[str]
        Numeric features (aggregated per group) to compare similarity.
    dims : tuple[str]
        Additional grouping keys beyond ad_id (e.g., ("objective",)).

    Returns
    -------
    dict with keys:
        - groups: per-group z-scored features and group_id
        - dist:   age-gender distribution (prop) per group_id
        - zcols:  names of z-score columns
        - zparams: mapping of log-feature -> (mean, std)
        - dims:   list of dims used
    """
    core_ag = drop_duplicate_columns(core_ag)
    gcols = ["ad_id"] + list(dims)

    # Age×Gender distribution per group
    dist = (core_ag.groupby(gcols + ["age", "gender"], dropna=False)["impressions"]
                    .sum()
                    .reset_index(name="impr"))
    dist["prop"] = dist.groupby(gcols, dropna=False)["impr"].transform(_safe_prop)
    dist = dist.drop(columns="impr")

    # Aggregate numeric features per group (sums)
    feats = (core_ag.groupby(gcols, dropna=False)[list(feature_cols)]
                    .sum(min_count=1)
                    .reset_index())

    # log1p then global Z-score
    for f in feature_cols:
        lf = f"_log_{f}"
        feats[lf] = np.log1p(pd.to_numeric(feats[f], errors="coerce").fillna(0.0))

    zs_params = {}
    for lf in [f"_log_{f}" for f in feature_cols]:
        m, s = _zscore_fit(feats[lf])
        zs_params[lf] = (m, s)
        feats[f"_z_{lf}"] = _zscore_apply(feats[lf], m, s)

    feats = feats.copy()
    feats["group_id"] = np.arange(len(feats))

    # Map distributions to groups
    dist = dist.merge(feats[gcols + ["group_id"]], on=gcols, how="inner")

    zcols = [c for c in feats.columns if c.startswith("_z_")]
    groups = feats[gcols + ["group_id"] + zcols + list(feature_cols)].copy()

    return {
        "groups": groups,
        "dist": dist,
        "zcols": zcols,
        "zparams": zs_params,
        "dims": list(dims),
    }


def _neighbors_weights(cart_g: pd.DataFrame,
                       neighbors_index: dict,
                       target_context: dict,
                       k: int = 5):
    """Compute weights via pooled distribution of k nearest groups (optional)."""
    if not neighbors_index:
        return None

    groups = neighbors_index["groups"]
    dist   = neighbors_index["dist"]
    zcols  = neighbors_index["zcols"]
    zparams= neighbors_index["zparams"]
    dims   = neighbors_index["dims"]

    # Restrict candidates by dims if present in context (e.g., objective)
    cand = groups.copy()
    for d in dims:
        if d in target_context:
            cand = cand[cand[d] == target_context[d]]
    if cand.empty:
        return None

    # Build target z-vector from provided features
    tfeat_raw = target_context.get("features", {})
    tvec = []
    for lf in [c.replace("_z_", "") for c in zcols]:
        base = lf.replace("_log_", "")
        val = float(tfeat_raw.get(base, 0.0))
        logv = np.log1p(max(val, 0.0))
        m, s = zparams[lf]
        tvec.append(_zscore_apply(logv, m, s))
    tvec = np.array(tvec, dtype=float) if len(tvec) else None

    if tvec is None or not len(cand):
        return None

    M = cand[zcols].to_numpy(dtype=float)
    dists = np.sqrt(np.sum((M - tvec[None, :])**2, axis=1))
    cand = cand.assign(_dist=dists).sort_values("_dist", ascending=True).head(k)

    # Pool neighbors' distributions and align to cart_g categories
    d_sub = dist[dist["group_id"].isin(cand["group_id"])][["group_id", "age", "gender", "prop"].copy()]
    base_keys = cart_g[["age", "gender"]].drop_duplicates()
    d_sub = base_keys.merge(d_sub, on=["age", "gender"], how="left")
    d_pooled = (d_sub.fillna({"prop": 0})
                    .groupby(["age", "gender"], dropna=False)["prop"]
                    .mean()
                    .reset_index(name="w"))

    w = d_pooled["w"].to_numpy(dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() <= 0:
        return None
    return w / w.sum()

# -----------------------------
# Weighting logic (present -> neighbors -> fallback -> uniform)
# -----------------------------

def _weights_age_gender(cart_g,
                        present_g: pd.DataFrame = None,
                        fallback_g: pd.DataFrame = None,
                        weight_col: str = "impressions",
                        strategy: str = "proportional",    # "proportional" | "neighbors"
                        neighbors_index: dict = None,
                        context: dict = None,
                        neighbors_k: int = 5):
    """
    Return weight vector for rows in cart_g based on available data.

    Priority:
      1) Distribution from present_g (actual data by age, gender).
      2) If strategy=="neighbors": pooled distribution from similar groups.
      3) Distribution from fallback_g (historical/aggregated).
      4) Uniform distribution.
    """
    key = ["age", "gender"]

    # 1) Present data
    if present_g is not None and not present_g.empty and weight_col in present_g:
        pg = present_g.groupby(key, dropna=False, as_index=False)[weight_col].sum()
        merged = cart_g[key].merge(pg, on=key, how="left")
        w = merged[weight_col].fillna(0).values
        if w.sum() > 0:
            return _safe_norm(w)

    # 2) Neighbors fallback (optional)
    if strategy == "neighbors" and neighbors_index is not None:
        w = _neighbors_weights(cart_g, neighbors_index, context or {}, k=neighbors_k)
        if w is not None and np.isfinite(w).all() and w.sum() > 0:
            return _safe_norm(w)

    # 3) Explicit fallback distribution
    if fallback_g is not None and not fallback_g.empty and weight_col in fallback_g:
        fb = fallback_g.groupby(key, dropna=False, as_index=False)[weight_col].sum()
        merged = cart_g[key].merge(fb, on=key, how="left")
        w = merged[weight_col].fillna(0).values
        if w.sum() > 0:
            return _safe_norm(w)

    # 4) Uniform
    return np.ones(len(cart_g)) / max(len(cart_g), 1)

# -----------------------------
# Core metrics estimation
# -----------------------------

def estimate_core_metrics(df_core,
                          cart,
                          metrics=("impressions", "clicks", "spend"),
                          dims=("objective", "age", "gender", "date_start", "date_stop"),
                          # new optional params
                          weighting_strategy: str = "proportional",  # or "neighbors"
                          neighbors_index: dict = None,
                          neighbors_k: int = 5):
    """
    Fill missing rows in core metrics so that totals per (ad_id, period, objective)
    match platform-level (age,gender=NA) targets.
    """
    try:
        logger.info("Starting core metrics estimation...")
        keys = ["ad_id"] + list(dims)
        df_core = drop_duplicate_columns(df_core)
        cart = drop_duplicate_columns(cart)

        core_cols = list(set(keys) | set(metrics))
        core = df_core[core_cols].copy()

        is_plat = core["age"].isna() & core["gender"].isna()
        core_plat = core[is_plat].copy()
        core_ag = core[~is_plat].copy()

        out = cart[keys].drop_duplicates().copy()

        present_keys = core_ag[keys].drop_duplicates()
        out = out.merge(present_keys.assign(_present=1), on=keys, how="left")
        missing_mask = out["_present"].isna()
        need_fill = out[missing_mask].drop(columns="_present")
        out = out.drop(columns="_present")

        out = out.merge(core_ag, on=keys, how="left")

        grp_keys = ["ad_id", "date_start", "date_stop", "objective"]
        hist = core_ag.groupby(["ad_id", "age", "gender"], dropna=False, as_index=False)["impressions"].sum()

        for gvals, cart_g in need_fill.groupby(grp_keys, dropna=False):
            ad_id, ds, de, obj = gvals

            plat_row = core_plat[
                (core_plat["ad_id"] == ad_id) &
                (core_plat["date_start"] == ds) &
                (core_plat["date_stop"] == de) &
                (core_plat["objective"] == obj)
            ]
            present_g = core_ag[
                (core_ag["ad_id"] == ad_id) &
                (core_ag["date_start"] == ds) &
                (core_ag["date_stop"] == de) &
                (core_ag["objective"] == obj)
            ]
            hist_g = hist[hist["ad_id"] == ad_id].rename(columns={"impressions": "weight"})

            # context features for neighbors (from platform-level row); only use available cols
            feat_ctx = {}
            for f in ("impressions", "clicks", "spend"):
                if f in plat_row:
                    feat_ctx[f] = float(pd.to_numeric(plat_row[f], errors="coerce").sum())
            ctx = {"objective": obj, "ad_id": ad_id, "features": feat_ctx}

            w = _weights_age_gender(
                cart_g,
                present_g=present_g,
                fallback_g=hist_g.rename(columns={"weight": "impressions"}),
                weight_col="impressions",
                strategy=weighting_strategy,
                neighbors_index=neighbors_index,
                context=ctx,
                neighbors_k=neighbors_k,
            )

            for m in metrics:
                target = float(plat_row[m].sum()) if (not plat_row.empty and m in plat_row) else np.nan
                observed = float(present_g[m].sum()) if (m in present_g) else 0.0

                if not np.isfinite(target) or abs(target - observed) < 1e-9:
                    continue

                residual = target - observed
                alloc = residual * w

                mask_insert = (
                    (out["ad_id"] == ad_id) &
                    (out["date_start"] == ds) &
                    (out["date_stop"] == de) &
                    (out["objective"] == obj) &
                    out["age"].isin(cart_g["age"]) &
                    out["gender"].isin(cart_g["gender"])
                )
                if m not in out.columns:
                    out[m] = 0.0
                out.loc[mask_insert & out[m].isna(), m] = 0.0
                targ_idx = out[mask_insert].sort_values(["age", "gender"]).index
                out.loc[targ_idx, m] = out.loc[targ_idx, m].fillna(0.0) + pd.Series(alloc, index=targ_idx)

        for m in metrics:
            if m in out.columns:
                out[m] = pd.to_numeric(out[m], errors="coerce").fillna(0.0)

        logger.info("Core metrics estimation completed.")
        return out
    except Exception as e:
        logger.exception(f"Error in estimate_core_metrics: {e}")
        raise

# -----------------------------
# Event table estimation
# -----------------------------

def _unique(seq):
    """Return a list preserving order while removing duplicates."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def estimate_event_table(df_event,
                         cart,
                         df_core_est,
                         dims=("objective", "age", "gender", "date_start", "date_stop"),
                         amount_col="amount",
                         id_cols=("ad_id", "action_type", "stat"),
                         weight_metric_for_split="impressions",
                         weighting_strategy="proportional",
                         neighbors_index=None,
                         neighbors_k=5):
    """
    Complete missing event rows so that totals per
    (ad_id, date_start, date_stop, objective, action_type, stat)
    match platform-level targets (age/gender = NA).
    Allocation across missing (age,gender) uses weights derived from core metrics
    or (optionally) a neighbors fallback.

    Parameters
    ----------
    df_event : DataFrame
        Event table with platform-level rows (age/gender NA) and possibly some age×gender rows.
    cart : DataFrame
        Cartesian product of (ad_id, dims), i.e. the full expected space of age×gender rows.
    df_core_est : DataFrame
        Core metrics (already estimated), used as weights (e.g., impressions) to split event totals.
    dims : tuple[str]
        Dimensions used as keys along with ad_id; must include 'age' and 'gender'.
    amount_col : str
        Name of the event amount column to balance (default 'amount').
    id_cols : tuple[str]
        Identifier columns for event type (default: ('ad_id','action_type','stat')).
    weight_metric_for_split : str
        Core metric used for splitting (default 'impressions').
    weighting_strategy : str
        'proportional' or 'neighbors' (neighbors used only when no present data in group).
    neighbors_index : dict | None
        Optional prebuilt neighbors index (see build_neighbors_index).
    neighbors_k : int
        Number of neighbors to pool when using neighbors fallback.

    Returns
    -------
    DataFrame
        Completed event table at age×gender granularity, with totals matching platform targets.
    """
    try:
        logger.info("Starting event table estimation...")

        # Ensure unique, ordered column lists
        keys = _unique(["ad_id"] + list(dims))
        id_cols = _unique(list(id_cols))

        # Ensure amount column exists
        if amount_col not in df_event.columns:
            df_event = df_event.copy()
            df_event[amount_col] = 0.0

        # Drop duplicated columns potentially created by previous merges
        df_event = drop_duplicate_columns(df_event)
        cart = drop_duplicate_columns(cart)
        df_core_est = drop_duplicate_columns(df_core_est)

        # Select only the necessary columns for events
        ev_cols = _unique(keys + id_cols + [amount_col])
        ev = df_event[ev_cols].copy()

        # Platform (no age/gender) vs age×gender rows
        is_plat = ev["age"].isna() & ev["gender"].isna()
        ev_plat = ev[is_plat].copy()
        ev_ag = ev[~is_plat].copy()

        # Base cartesian space: (ad_id, dims) x (action_type, stat) per ad
        base = cart[keys].drop_duplicates().copy()
        act_stat = ev[id_cols].drop_duplicates()

        # Align dtypes for categorical keys to avoid object/float mismatches
        for col in ("age", "gender"):
            if col in ev.columns:
                ev[col] = ev[col].astype(object)
            if col in ev_ag.columns:
                ev_ag[col] = ev_ag[col].astype(object)
            if col in base.columns:
                base[col] = base[col].astype(object)

        full_space = base.merge(act_stat, on=["ad_id"], how="inner")

        # Join keys used to match presence at age×gender level
        join_cols = _unique(keys + id_cols)

        # Make sure presence keys have compatible dtypes
        present_keys = ev_ag[join_cols].drop_duplicates()
        for col in ("age", "gender"):
            if col in present_keys.columns:
                present_keys[col] = present_keys[col].astype(object)

        # Missing combinations (need fill) = in full_space but not in ev_ag
        to_fill = full_space.merge(present_keys.assign(_p=1), on=join_cols, how="left")
        to_fill = to_fill[to_fill["_p"].isna()].drop(columns="_p")

        # Start output from existing rows at age×gender level (left join to retain full_space keys)
        out = full_space.merge(ev_ag, on=join_cols, how="left")

        # Grouping keys for target balancing
        grp_keys = ["ad_id", "date_start", "date_stop", "objective", "action_type", "stat"]

        # Core weights available for splitting
        core_w = df_core_est[keys + [weight_metric_for_split]].copy()

        for gvals, cart_g in to_fill.groupby(grp_keys, dropna=False):
            ad_id, ds, de, obj, act, stat = gvals

            # Platform-level target (age/gender NA)
            plat_row = ev_plat[
                (ev_plat["ad_id"] == ad_id) &
                (ev_plat["date_start"] == ds) &
                (ev_plat["date_stop"] == de) &
                (ev_plat["objective"] == obj) &
                (ev_plat["action_type"] == act) &
                (ev_plat["stat"] == stat)
            ]
            # Observed age×gender rows in this group
            present_g = ev_ag[
                (ev_ag["ad_id"] == ad_id) &
                (ev_ag["date_start"] == ds) &
                (ev_ag["date_stop"] == de) &
                (ev_ag["objective"] == obj) &
                (ev_ag["action_type"] == act) &
                (ev_ag["stat"] == stat)
            ]

            target = float(plat_row[amount_col].sum()) if not plat_row.empty else np.nan
            if not np.isfinite(target):
                continue

            observed = float(present_g[amount_col].sum()) if (amount_col in present_g) else 0.0
            if abs(target - observed) < 1e-9:
                # Already balanced
                continue

            # Slice core weights for this (ad_id, period, objective)
            core_slice = core_w[
                (core_w["ad_id"] == ad_id) &
                (core_w["date_start"] == ds) &
                (core_w["date_stop"] == de) &
                (core_w["objective"] == obj)
            ][["age", "gender", weight_metric_for_split]]

            # Compute weights over the missing (age,gender)
            w = _weights_age_gender(
                cart_g[["age", "gender"]],
                present_g=None,  # No use of event distribution to avoid leakage
                fallback_g=core_slice.rename(columns={weight_metric_for_split: "impressions"}),
                weight_col="impressions",
                strategy=weighting_strategy,
                neighbors_index=neighbors_index,
                context={"objective": obj, "ad_id": ad_id, "features": {}},
                neighbors_k=neighbors_k,
            )

            # Insert allocation into output rows matching the missing combos
            mask_insert = (
                (out["ad_id"] == ad_id) &
                (out["date_start"] == ds) &
                (out["date_stop"] == de) &
                (out["objective"] == obj) &
                (out["action_type"] == act) &
                (out["stat"] == stat) &
                out["age"].isin(cart_g["age"]) &
                out["gender"].isin(cart_g["gender"])
            )
            if amount_col not in out.columns:
                out[amount_col] = 0.0
            out.loc[mask_insert & out[amount_col].isna(), amount_col] = 0.0

            # Align allocation vector with the target indices (sorted by age,gender)
            targ_idx = out[mask_insert].sort_values(["age", "gender"]).index
            out.loc[targ_idx, amount_col] = (
                out.loc[targ_idx, amount_col].fillna(0.0) + pd.Series(w, index=targ_idx) * (target - observed)
            )

        out[amount_col] = pd.to_numeric(out[amount_col], errors="coerce").fillna(0.0)
        logger.info("Event table estimation completed.")
        return out

    except Exception as e:
        logger.exception(f"Error in estimate_event_table: {e}")
        raise

def build_neighbors_index(core_ag: pd.DataFrame = None,
                          df: pd.DataFrame = None,
                          feature_cols=("impressions", "clicks", "spend"),
                          dims=("objective",)):
    if core_ag is None and df is not None:
        core_ag = df
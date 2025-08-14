# tests/test_cartesian_missing.py
import pandas as pd
import numpy as np
import pytest

from src.cartesian_missing import (
    DIMS,
    domains_per_ad,
    build_cartesian,
    find_missing,
    build_cartesian_for_ad,
)


def _expected_count_per_ad(dom):
    """Product of domain lengths for the categorical dims."""
    lengths = [len(dom[d]) for d in DIMS]
    return int(np.prod(lengths, dtype=np.int64))


# ----------------------------
# Core size / product checks
# ----------------------------

def test_cartesian_count_matches_domain_product_small():
    """
    Synthetic dataset:
    ad 1: obj={conv}, plat={fb, ig}, age={18-24,25-34}, gender={male,female} -> 1*2*2*2 = 8
    ad 2: obj={conv,reach}, plat={fb}, age={NA->domain=[NA]}, gender={male} -> 2*1*1*1 = 2
    """
    df_core = pd.DataFrame([
        # ad 1
        {"ad_id": 1, "objective": "conversions", "publisher_platform": "facebook",  "age": "18-24", "gender": "male"},
        {"ad_id": 1, "objective": "conversions", "publisher_platform": "instagram", "age": "25-34", "gender": "female"},
        # ad 2
        {"ad_id": 2, "objective": "conversions", "publisher_platform": "facebook", "age": pd.NA,    "gender": "male"},
        {"ad_id": 2, "objective": "reach",       "publisher_platform": "facebook", "age": pd.NA,    "gender": "male"},
    ])

    doms = domains_per_ad(df_core)
    cart = build_cartesian(df_core)

    expected = {
        1: _expected_count_per_ad(doms[1]),
        2: _expected_count_per_ad(doms[2]),
    }
    got = cart.groupby("ad_id").size().to_dict()

    assert got[1] == expected[1]
    assert got[2] == expected[2]
    assert len(cart) == sum(expected.values())


def test_cartesian_count_expands_with_extra_sources():
    """
    Extra sources can expand per-ad domains.
    Example: core has only platform=facebook, extra shows instagram for same ad -> domain doubles.
    """
    df_core = pd.DataFrame([
        {"ad_id": 3, "objective": "conversions", "publisher_platform": "facebook", "age": "18-24", "gender": "male"},
    ])
    extra = pd.DataFrame([
        {"ad_id": 3, "publisher_platform": "instagram"},
    ])

    doms_no_extra = domains_per_ad(df_core)
    expected_no_extra = _expected_count_per_ad(doms_no_extra[3])

    doms_with_extra = domains_per_ad(df_core, extra_sources=[extra])
    expected_with_extra = _expected_count_per_ad(doms_with_extra[3])

    cart_with_extra = build_cartesian(df_core, extra_sources=[extra])

    assert expected_with_extra == expected_no_extra * 2
    assert len(cart_with_extra[cart_with_extra["ad_id"] == 3]) == expected_with_extra


def test_cartesian_count_when_all_dims_are_na_for_ad():
    """
    If an ad has only NA across all dims, each domain falls back to [NA] => size 1.
    """
    df_core = pd.DataFrame([
        {"ad_id": 4, "objective": pd.NA, "publisher_platform": pd.NA, "age": pd.NA, "gender": pd.NA},
    ])
    doms = domains_per_ad(df_core)
    cart = build_cartesian(df_core)

    assert doms[4]["objective"] == [pd.NA]
    assert doms[4]["publisher_platform"] == [pd.NA]
    assert doms[4]["age"] == [pd.NA]
    assert doms[4]["gender"] == [pd.NA]
    assert len(cart[cart["ad_id"] == 4]) == 1  # 1*1*1*1


@pytest.mark.skipif(not __import__("os").path.exists("artifacts/stage1_core.csv"),
                    reason="stage1_core.csv not found; skipping integration count test")
def test_cartesian_total_count_matches_domains_on_real_core_if_exists():
    """
    Integration sanity: total cartesian rows equals the sum of per-ad products.
    """
    import os
    df_core = pd.read_csv(os.path.join("artifacts", "stage1_core.csv"))
    doms = domains_per_ad(df_core)

    expected_total = 0
    for _, dom in doms.items():
        expected_total += int(np.prod([len(dom[d]) for d in DIMS], dtype=np.int64))

    cart = build_cartesian(df_core)
    assert len(cart) == expected_total


# ----------------------------
# Normalization & allowlist
# ----------------------------

def test_platform_alias_and_allowlist_normalization():
    """
    'fb' -> 'facebook', 'audience network' -> 'audience_network', and 'unknown' is dropped (NA).
    Only allowlisted platforms should appear in the per-ad domain.
    """
    df_core = pd.DataFrame([
        {"ad_id": 10, "objective": "conversions", "publisher_platform": "fb",               "age": "18-24", "gender": "male"},
        {"ad_id": 10, "objective": "conversions", "publisher_platform": "audience network", "age": "25-34", "gender": "female"},
        {"ad_id": 10, "objective": "conversions", "publisher_platform": "unknown",          "age": "25-34", "gender": "female"},
    ])

    doms = domains_per_ad(df_core)
    pp_vals = set(doms[10]["publisher_platform"])

    assert "facebook" in pp_vals
    assert "audience_network" in pp_vals
    # 'unknown' should NOT remain as a concrete value in the domain
    assert "unknown" not in pp_vals


def test_invalid_platform_falls_back_to_na_domain():
    """
    If all observed platforms are invalid (e.g., 'snapchat'), the platform domain should be [NA].
    That yields a cartesian size of 1 when all other dims are also NA or missing.
    """
    df_core = pd.DataFrame([
        {"ad_id": 11, "publisher_platform": "snapchat", "objective": pd.NA, "age": pd.NA, "gender": pd.NA},
    ])
    doms = domains_per_ad(df_core)
    assert doms[11]["publisher_platform"] == [pd.NA]
    assert doms[11]["objective"] == [pd.NA]
    assert doms[11]["age"] == [pd.NA]
    assert doms[11]["gender"] == [pd.NA]

    cart = build_cartesian(df_core)
    assert len(cart[cart["ad_id"] == 11]) == 1  # 1*1*1*1


def test_extra_source_with_alias_expands_domain():
    """
    Extra source that uses an alias (e.g., 'fb') should expand the platform domain properly to 'facebook'.
    """
    df_core = pd.DataFrame([
        {"ad_id": 12, "objective": "conversions", "publisher_platform": "instagram", "age": "18-24", "gender": "male"},
    ])
    extra = pd.DataFrame([
        {"ad_id": 12, "publisher_platform": "fb"},
    ])

    doms = domains_per_ad(df_core, extra_sources=[extra])
    pp_vals = set(doms[12]["publisher_platform"])
    assert pp_vals == {"instagram", "facebook"}

    cart = build_cartesian(df_core, extra_sources=[extra])
    assert len(cart[cart["ad_id"] == 12]) == _expected_count_per_ad(doms[12])


# ----------------------------
# Missing combos (NA wildcard)
# ----------------------------

def test_find_missing_with_wildcard_rows():
    """
    A row with NA on some dims acts as a wildcard and covers the entire ad-specific domain on those dims.
    Here, two explicit Instagram rows already cover 2 combos, so only 2 should be missing.
    """
    df_core = pd.DataFrame([
        {"ad_id": 20, "objective": "conversions", "publisher_platform": "facebook",  "age": pd.NA,    "gender": pd.NA},
        {"ad_id": 20, "objective": "conversions", "publisher_platform": "instagram", "age": "18-24", "gender": "male"},
        {"ad_id": 20, "objective": "conversions", "publisher_platform": "instagram", "age": "25-34", "gender": "female"},
    ])

    doms = domains_per_ad(df_core)
    cart = build_cartesian(df_core)
    missing = find_missing(df_core, cart)

    # Full size is 1*2*2*2 = 8
    assert len(cart[cart["ad_id"] == 20]) == _expected_count_per_ad(doms[20])

    # Only two instagram combos remain uncovered:
    missing_20 = missing[missing["ad_id"] == 20]
    assert len(missing_20) == 2
    got = set(map(tuple, missing_20[["publisher_platform","age","gender"]].itertuples(index=False, name=None)))
    expected = {
        ("instagram", "18-24", "female"),
        ("instagram", "25-34", "male"),
    }
    assert got == expected


# ----------------------------
# Per-ad helpers
# ----------------------------

def test_build_cartesian_for_ad_returns_exact_grid():
    """
    build_cartesian_for_ad should return the per-ad cartesian grid equal to the per-ad product.
    """
    df_core = pd.DataFrame([
        {"ad_id": 30, "objective": "conversions", "publisher_platform": "instagram", "age": "18-24", "gender": "male"},
        {"ad_id": 30, "objective": "conversions", "publisher_platform": "facebook",  "age": "25-34", "gender": "female"},
    ])

    doms = domains_per_ad(df_core)
    from src.cartesian_missing import build_cartesian_for_ad  # local import to avoid circulars in some runners
    grid = build_cartesian_for_ad(df_core, 30)

    assert len(grid) == _expected_count_per_ad(doms[30])
    assert set(grid["publisher_platform"]) == {"facebook", "instagram"}
    assert set(grid["age"]) == {"18-24", "25-34"}
    assert set(grid["gender"]) == {"male", "female"}
    assert set(grid["objective"]) == {"conversions"}


def test_empty_core_yields_empty_cartesian():
    """Empty input should produce an empty cartesian DataFrame with the right columns."""
    df_core = pd.DataFrame(columns=["ad_id"] + DIMS)
    cart = build_cartesian(df_core)
    assert cart.empty
    assert list(cart.columns) == ["ad_id"] + DIMS


def test_find_missing_with_wildcard_rows_expect_four_when_instagram_only_in_domain():
    """
    Domain includes Instagram via extra source, but no core rows cover Instagram combos,
    so 4 combos (instagram x 2 ages x 2 genders) are missing.
    """
    df_core = pd.DataFrame([
        {"ad_id": 21, "objective": "conversions", "publisher_platform": "facebook", "age": pd.NA, "gender": pd.NA},
    ])
    extra = pd.DataFrame([
        {"ad_id": 21, "publisher_platform": "instagram"},  # expands domain only
        {"ad_id": 21, "age": "18-24"},
        {"ad_id": 21, "age": "25-34"},
        {"ad_id": 21, "gender": "male"},
        {"ad_id": 21, "gender": "female"},
    ])

    cart = build_cartesian(df_core, extra_sources=[extra])
    missing = find_missing(df_core, cart, extra_sources=[extra])
    missing_21 = missing[missing["ad_id"] == 21]

    assert len(missing_21) == 4
    assert set(missing_21["publisher_platform"]) == {"instagram"}
    assert set(missing_21["age"]) == {"18-24", "25-34"}
    assert set(missing_21["gender"]) == {"male", "female"}

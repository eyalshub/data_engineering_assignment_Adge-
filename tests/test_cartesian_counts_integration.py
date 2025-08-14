# tests/test_cartesian_counts_integration.py
import os
import numpy as np
import pandas as pd
import pytest

from src.cartesian_missing import domains_per_ad, build_cartesian, DIMS

def _domain_product(dom: dict) -> int:
    return int(np.prod([len(dom[d]) for d in DIMS], dtype=np.int64))

@pytest.mark.integration
def test_cartesian_total_count_matches_domains_on_real_core_if_exists():
    core_p = os.path.join("artifacts", "stage1_core.csv")
    if not os.path.exists(core_p):
        pytest.skip("stage1_core.csv not found; skipping integration count test")

    core = pd.read_csv(core_p)

    # 1) Core only
    doms = domains_per_ad(core)
    exp = int(sum(_domain_product(dom) for dom in doms.values()))
    cart = build_cartesian(core)
    assert len(cart) == exp

    # 2) With events if present
    event_names = [
        "stage1_events_actions.csv",
        "stage1_events_action_values.csv",
        # add more if needed:
        # "stage1_events_video_play_actions.csv",
        # "stage1_events_video_thruplay_watched_actions.csv",
        # "stage1_events_video_avg_time_watched_actions.csv",
    ]
    event_paths = [os.path.join("artifacts", n) for n in event_names if os.path.exists(os.path.join("artifacts", n))]
    if event_paths:
        events = [pd.read_csv(p) for p in event_paths]
        doms_all = domains_per_ad(core, extra_sources=events)
        exp_all = int(sum(_domain_product(dom) for dom in doms_all.values()))
        cart_all = build_cartesian(core, extra_sources=events)
        assert len(cart_all) == exp_all

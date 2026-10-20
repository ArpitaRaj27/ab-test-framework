"""Tests for ab_test_template.py — run with:  pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ab_test_template import analyze, detect_metric_type, run_test  # noqa: E402

rng = np.random.default_rng(123)


def make_df(control_vals, treatment_vals, group_col="variant", metric="m"):
    return pd.DataFrame({
        group_col: ["A"] * len(control_vals) + ["B"] * len(treatment_vals),
        metric: np.concatenate([control_vals, treatment_vals]),
    })


def test_large_binary_uses_z_test():
    df = make_df(rng.binomial(1, 0.10, 5000), rng.binomial(1, 0.13, 5000))
    r = run_test(df, "variant", "m", "A", "B")
    assert r.metric_type == "binary"
    assert r.test == "Two-proportion z-test"
    assert r.effect_size_name == "Cohen's h"
    assert r.ci_low < r.abs_diff < r.ci_high


def test_tiny_binary_uses_fisher():
    df = make_df(rng.binomial(1, 0.05, 30), rng.binomial(1, 0.10, 30))
    r = run_test(df, "variant", "m", "A", "B")
    assert r.test == "Fisher's exact test"
    assert any("small sample" in w for w in r.warnings)


def test_large_continuous_uses_welch():
    df = make_df(rng.normal(50, 10, 2000), rng.normal(51, 10, 2000))
    r = run_test(df, "variant", "m", "A", "B")
    assert r.test == "Welch's t-test"
    assert r.effect_size_name == "Cohen's d"


def test_small_skewed_continuous_uses_mann_whitney():
    ctrl = np.where(rng.random(150) < 0.9, 0, rng.exponential(80, 150))
    trt = np.where(rng.random(150) < 0.85, 0, rng.exponential(80, 150))
    r = run_test(make_df(ctrl, trt), "variant", "m", "A", "B")
    assert r.test == "Mann-Whitney U"
    assert r.effect_size_name == "rank-biserial r"
    assert -1 <= r.effect_size <= 1


def test_detects_true_effect_and_null():
    big = make_df(rng.binomial(1, 0.10, 20000), rng.binomial(1, 0.12, 20000))
    assert run_test(big, "variant", "m", "A", "B").significant
    same = make_df(rng.binomial(1, 0.10, 20000), rng.binomial(1, 0.10, 20000))
    # Not guaranteed for every seed, but with this fixed RNG the null case should not reject
    assert not run_test(same, "variant", "m", "A", "B").significant


def test_underpowered_warning():
    df = make_df(rng.binomial(1, 0.05, 1000), rng.binomial(1, 0.055, 1000))
    r = run_test(df, "variant", "m", "A", "B", mde=0.10)
    assert r.power_at_mde < 0.8
    assert any("UNDERPOWERED" in w for w in r.warnings)


def test_srm_failure_is_flagged():
    df = make_df(rng.binomial(1, 0.1, 5000), rng.binomial(1, 0.1, 4500))
    table = analyze(df, "variant", ["m"], "A", "B", verbose=False)
    assert table.loc["m", "warnings"].startswith("SRM FAIL")


def test_missing_values_are_dropped_with_note():
    df = make_df(rng.normal(10, 2, 500), rng.normal(10, 2, 500))
    df.loc[:9, "m"] = np.nan
    r = run_test(df, "variant", "m", "A", "B")
    assert r.n_control == 490
    assert any("dropped 10 rows" in w for w in r.warnings)


def test_rejects_wrong_number_of_groups():
    df = pd.DataFrame({"variant": ["A", "B", "C"] * 10, "m": rng.binomial(1, 0.5, 30)})
    with pytest.raises(ValueError, match="exactly 2 groups"):
        analyze(df, "variant", ["m"], "A", "B", verbose=False)


def test_rejects_unknown_group_label():
    df = make_df(rng.binomial(1, 0.1, 100), rng.binomial(1, 0.1, 100))
    with pytest.raises(ValueError, match="not found"):
        analyze(df, "variant", ["m"], "A", "Z", verbose=False)


def test_metric_type_detection():
    assert detect_metric_type(pd.Series([0, 1, 1, 0])) == "binary"
    assert detect_metric_type(pd.Series([True, False])) == "binary"
    assert detect_metric_type(pd.Series([1, 2, 3, 4, 5])) == "continuous"


def test_matches_freshcrate_notebook_results():
    data = Path(__file__).resolve().parents[1] / "data" / "ab_test_clean.csv"
    if not data.exists():
        pytest.skip("run src/generate_data.py and the notebook first")
    df = pd.read_csv(data)
    table = analyze(df, "group", ["order_placed", "revenue"], "control", "treatment", mde=0.10, verbose=False)
    assert table.loc["order_placed", "p_value"] == pytest.approx(0.00022, abs=1e-5)
    assert table.loc["order_placed", "rel_lift"] == pytest.approx(0.1714, abs=1e-3)
    assert table.loc["revenue", "test"] == "Welch's t-test"


def test_ordinal_metric_forces_mann_whitney():
    df = make_df(rng.integers(1, 6, 2000), rng.integers(2, 6, 2000))
    table = analyze(df, "variant", ["m"], "A", "B", ordinal=["m"], verbose=False)
    assert table.loc["m", "test"] == "Mann-Whitney U"
    assert "ordinal" in table.loc["m", "reason"]

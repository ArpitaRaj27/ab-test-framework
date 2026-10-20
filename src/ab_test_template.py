"""
ab_test_template.py — reusable two-group A/B test analysis.

Drop in any dataset with one row per user, a group column, and one or more
metric columns. The template:
    1. validates the input (exactly two groups, missing values, group balance)
    2. detects each metric's type (binary vs. continuous)
    3. picks the right test automatically and says why
    4. returns one clean summary table: rates/means, lift, 95% CI, p-value,
       effect size, power, required sample size, and warnings

Test selection logic
--------------------
    binary (0/1)  -> two-proportion z-test
                     Fisher's exact test if any 2x2 cell < 10
    continuous    -> Welch's t-test if n > 25 * skewness^2  (mean is ~normal by CLT)
                     Mann-Whitney U otherwise (rank-based, robust to skew)
    ordinal       -> Mann-Whitney U (ratings, NPS; pass them via `ordinal=` / --ordinal)

Python usage
------------
    import pandas as pd
    from ab_test_template import analyze

    df = pd.read_csv("my_test.csv")
    table = analyze(df, group_col="variant", metrics=["converted", "revenue"],
                    control="A", treatment="B", mde=0.05)
    print(table)

Command line
------------
    python src/ab_test_template.py data/ab_test_clean.csv \\
        --group group --control control --treatment treatment \\
        --metrics order_placed revenue --mde 0.10
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_effectsize,
    proportions_ztest,
)

MIN_CELL = 10            # smallest 2x2 cell for the z-test's normal approximation
SRM_ALPHA = 0.01         # sample-ratio-mismatch threshold
TARGET_POWER = 0.80
N_BOOT = 5000


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class TestResult:
    metric: str
    metric_type: str
    test: str
    reason: str
    n_control: int
    n_treatment: int
    control_value: float
    treatment_value: float
    abs_diff: float
    rel_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    effect_size_name: str
    effect_size: float
    power_at_mde: Optional[float]
    n_needed_per_group: Optional[int]
    significant: bool
    warnings: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "metric": self.metric,
            "type": self.metric_type,
            "test": self.test,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "control": self.control_value,
            "treatment": self.treatment_value,
            "abs_diff": self.abs_diff,
            "rel_lift": self.rel_lift,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "effect_size": f"{self.effect_size_name} = {self.effect_size:.4f}",
            "power_at_mde": self.power_at_mde,
            "n_needed_per_group": self.n_needed_per_group,
            "significant": self.significant,
            "reason": self.reason,
            "warnings": "; ".join(self.warnings) if self.warnings else "",
        }


# --------------------------------------------------------------------------
# Step 1: validation and test selection
# --------------------------------------------------------------------------
def srm_check(n_control: int, n_treatment: int, expected_share: float = 0.5) -> float:
    """Chi-square goodness-of-fit p-value for the designed split."""
    total = n_control + n_treatment
    expected = [total * (1 - expected_share), total * expected_share]
    return stats.chisquare([n_control, n_treatment], f_exp=expected).pvalue


def detect_metric_type(values: pd.Series) -> str:
    uniq = set(pd.unique(values.dropna()))
    return "binary" if uniq <= {0, 1, True, False} else "continuous"


def select_test(a: np.ndarray, b: np.ndarray, metric_type: str) -> tuple[str, str]:
    """Return (test name, plain-English reason)."""
    if metric_type == "ordinal":
        return "Mann-Whitney U", "ordinal scale (e.g. 1-5 rating): gaps between levels aren't equal, so compare ranks"
    if metric_type == "binary":
        cells = [a.sum(), len(a) - a.sum(), b.sum(), len(b) - b.sum()]
        smallest = int(min(cells))
        if smallest < MIN_CELL:
            return "Fisher's exact test", f"binary metric; smallest 2x2 cell = {smallest} < {MIN_CELL}"
        return "Two-proportion z-test", f"binary metric; smallest 2x2 cell = {smallest:,} >= {MIN_CELL}"

    skew = max(abs(stats.skew(a)), abs(stats.skew(b)))
    n_needed = int(np.ceil(25 * skew ** 2))
    n_min = min(len(a), len(b))
    if n_min > n_needed:
        return "Welch's t-test", f"continuous; skew {skew:.2f} needs n > {n_needed:,} for a normal mean, have {n_min:,}"
    return "Mann-Whitney U", f"continuous; skew {skew:.2f} needs n > {n_needed:,}, only {n_min:,}"


# --------------------------------------------------------------------------
# Step 2: the tests
# --------------------------------------------------------------------------
def _binary_test(a, b, test, alpha, mde):
    xc, nc, xt, nt = a.sum(), len(a), b.sum(), len(b)
    pc, pt = xc / nc, xt / nt

    if test == "Fisher's exact test":
        p = stats.fisher_exact([[xt, nt - xt], [xc, nc - xc]])[1]
        lo, hi = confint_proportions_2indep(xt, nt, xc, nc, method="newcomb", compare="diff", alpha=alpha)
    else:
        p = proportions_ztest([xt, xc], [nt, nc])[1]
        lo, hi = confint_proportions_2indep(xt, nt, xc, nc, method="wald", compare="diff", alpha=alpha)

    h = proportion_effectsize(pt, pc)
    power = n_needed = None
    if mde is not None and 0 < pc < 1:
        h_mde = proportion_effectsize(min(pc * (1 + mde), 0.9999), pc)
        calc = NormalIndPower()
        power = calc.power(effect_size=h_mde, nobs1=nc, alpha=alpha, ratio=nt / nc)
        n_needed = int(np.ceil(calc.solve_power(effect_size=h_mde, alpha=alpha, power=TARGET_POWER, ratio=1.0)))
    return pc, pt, lo, hi, p, "Cohen's h", h, power, n_needed


def _continuous_test(a, b, test, alpha, mde, rng):
    mc, mt = a.mean(), b.mean()
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))

    if test == "Welch's t-test":
        p = stats.ttest_ind(b, a, equal_var=False).pvalue
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt(va + vb)
        dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        t_crit = stats.t.ppf(1 - alpha / 2, dof)
        lo, hi = (mt - mc) - t_crit * se, (mt - mc) + t_crit * se
        es_name, es = "Cohen's d", (mt - mc) / pooled_sd if pooled_sd > 0 else 0.0
    else:
        u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
        # Bootstrap CI for the difference in means (no distribution assumption)
        diffs = np.array([rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        # Rank-biserial correlation: the effect size that matches Mann-Whitney
        es_name, es = "rank-biserial r", 2 * u / (len(a) * len(b)) - 1

    power = n_needed = None
    if mde is not None and pooled_sd > 0 and mc != 0:
        d_mde = abs(mc * mde) / pooled_sd
        calc = TTestIndPower()
        power = calc.power(effect_size=d_mde, nobs1=len(a), alpha=alpha, ratio=len(b) / len(a))
        n_needed = int(np.ceil(calc.solve_power(effect_size=d_mde, alpha=alpha, power=TARGET_POWER, ratio=1.0)))
    return mc, mt, lo, hi, p, es_name, es, power, n_needed


# --------------------------------------------------------------------------
# Step 3: public API
# --------------------------------------------------------------------------
def run_test(df: pd.DataFrame, group_col: str, metric: str, control, treatment,
             alpha: float = 0.05, mde: Optional[float] = None,
             metric_type: str = "auto", seed: int = 0) -> TestResult:
    """Analyze one metric. `mde` is the minimum detectable effect as a relative lift (0.10 = 10%)."""
    notes = []
    sub = df[[group_col, metric]]
    n_missing = int(sub[metric].isna().sum())
    if n_missing:
        notes.append(f"dropped {n_missing:,} rows with missing {metric}")
    sub = sub.dropna(subset=[metric])

    a = sub.loc[sub[group_col] == control, metric].astype(float).to_numpy()
    b = sub.loc[sub[group_col] == treatment, metric].astype(float).to_numpy()
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"need at least 2 rows per group for '{metric}' (got {len(a)} and {len(b)})")

    mtype = detect_metric_type(sub[metric]) if metric_type == "auto" else metric_type
    test, reason = select_test(a, b, mtype)

    if mtype == "binary":
        cv, tv, lo, hi, p, es_name, es, power, n_needed = _binary_test(a, b, test, alpha, mde)
    else:  # continuous or ordinal
        rng = np.random.default_rng(seed)
        cv, tv, lo, hi, p, es_name, es, power, n_needed = _continuous_test(a, b, test, alpha, mde, rng)

    if power is not None and power < TARGET_POWER:
        notes.append(f"UNDERPOWERED: {power:.0%} power for a {mde:.0%} lift; need {n_needed:,}/group")
    if min(len(a), len(b)) < 100:
        notes.append("small sample (<100 per group): treat results as directional")

    return TestResult(
        metric=metric, metric_type=mtype, test=test, reason=reason,
        n_control=len(a), n_treatment=len(b),
        control_value=cv, treatment_value=tv, abs_diff=tv - cv,
        rel_lift=(tv / cv - 1) if cv != 0 else np.nan,
        ci_low=lo, ci_high=hi, p_value=p,
        effect_size_name=es_name, effect_size=es,
        power_at_mde=power, n_needed_per_group=n_needed,
        significant=bool(p < alpha), warnings=notes,
    )


def analyze(df: pd.DataFrame, group_col: str, metrics: Sequence[str],
            control=None, treatment=None, alpha: float = 0.05,
            mde: Optional[float] = None, expected_split: float = 0.5,
            ordinal: Sequence[str] = (), verbose: bool = True) -> pd.DataFrame:
    """Analyze several metrics at once and return one summary table.

    `ordinal` lists metrics on a ranked scale (ratings, NPS); they always use Mann-Whitney U.
    """
    groups = list(pd.unique(df[group_col].dropna()))
    if len(groups) != 2:
        raise ValueError(f"'{group_col}' must have exactly 2 groups, found {len(groups)}: {groups}")
    if control is None or treatment is None:
        control, treatment = sorted(groups, key=str)
        warnings.warn(f"control/treatment not given; using control='{control}', treatment='{treatment}'")
    missing = {control, treatment} - set(groups)
    if missing:
        raise ValueError(f"group labels {missing} not found in '{group_col}' (found {groups})")

    n_c = int((df[group_col] == control).sum())
    n_t = int((df[group_col] == treatment).sum())
    p_srm = srm_check(n_c, n_t, expected_split)

    results = [run_test(df, group_col, m, control, treatment, alpha, mde,
                        metric_type="ordinal" if m in ordinal else "auto") for m in metrics]
    table = pd.DataFrame([r.as_row() for r in results]).set_index("metric")
    if p_srm < SRM_ALPHA:
        table["warnings"] = ("SRM FAIL (p={:.2g}): group sizes don't match the designed split; "
                             "fix assignment before trusting results. ".format(p_srm) + table["warnings"])

    if verbose:
        print_report(table, control, treatment, n_c, n_t, p_srm, alpha, mde)
    return table


# --------------------------------------------------------------------------
# Step 4: human-readable report
# --------------------------------------------------------------------------
def _fmt(v, mtype):
    return f"{v:.2%}" if mtype == "binary" else f"{v:,.3f}"


def print_report(table: pd.DataFrame, control, treatment, n_c, n_t, p_srm, alpha, mde) -> None:
    line = "=" * 78
    print(line)
    print(f"A/B TEST REPORT   control = '{control}'  vs  treatment = '{treatment}'")
    print(line)
    print(f"Sample sizes: {n_c:,} control / {n_t:,} treatment   "
          f"(treatment share {n_t / (n_c + n_t):.2%})")
    srm = "PASS" if p_srm >= SRM_ALPHA else "FAIL — investigate assignment"
    print(f"Sample ratio check: p = {p_srm:.4f}  -> {srm}")
    print(f"Significance level: {alpha}   Two-tailed   "
          f"MDE: {'not set' if mde is None else f'{mde:.0%} relative'}\n")

    for metric, r in table.iterrows():
        t = r["type"]
        ci = (f"[{r['ci_low'] * 100:+.2f}, {r['ci_high'] * 100:+.2f}] pts" if t == "binary"
              else f"[{r['ci_low']:+,.3f}, {r['ci_high']:+,.3f}]")
        print(f"-- {metric}  ({t})")
        print(f"   Test chosen:   {r['test']}   because {r['reason']}")
        print(f"   Control:       {_fmt(r['control'], t)}     Treatment: {_fmt(r['treatment'], t)}")
        diff = f"{r['abs_diff'] * 100:+.2f} pts" if t == "binary" else f"{r['abs_diff']:+,.3f}"
        p_txt = "<0.00001" if r["p_value"] < 1e-5 else f"{r['p_value']:.5f}"
        print(f"   Difference:    {diff} ({r['rel_lift']:+.1%} relative)   95% CI {ci}")
        print(f"   p-value:       {p_txt}  -> "
              f"{'SIGNIFICANT' if r['significant'] else 'not significant'} at alpha = {alpha}")
        print(f"   Effect size:   {r['effect_size']}")
        if pd.notna(r["power_at_mde"]):
            print(f"   Power at MDE:  {r['power_at_mde']:.0%}   (80% needs {int(r['n_needed_per_group']):,} per group)")
        if r["warnings"]:
            print(f"   ! {r['warnings']}")
        print()

    summary = table[["test", "control", "treatment", "rel_lift", "ci_low", "ci_high",
                     "p_value", "effect_size", "power_at_mde", "significant"]].copy()
    print("SUMMARY TABLE")
    print(summary.to_string(float_format=lambda v: f"{v:,.4f}"))
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a two-group A/B test on a CSV file.")
    parser.add_argument("csv", help="path to a CSV with one row per user")
    parser.add_argument("--group", required=True, help="column holding the group label")
    parser.add_argument("--metrics", nargs="+", required=True, help="metric column(s) to test")
    parser.add_argument("--control", help="label of the control group")
    parser.add_argument("--treatment", help="label of the treatment group")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--mde", type=float, default=None, help="minimum detectable effect, relative (0.10 = 10%%)")
    parser.add_argument("--split", type=float, default=0.5, help="designed treatment share (default 0.5)")
    parser.add_argument("--ordinal", nargs="*", default=[], help="metrics on a ranked scale (ratings); uses Mann-Whitney")
    parser.add_argument("--out", help="optional path to save the summary table as CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    table = analyze(df, args.group, args.metrics, args.control, args.treatment,
                    alpha=args.alpha, mde=args.mde, expected_split=args.split, ordinal=args.ordinal)
    if args.out:
        table.to_csv(args.out)
        print(f"Saved summary table -> {args.out}")


if __name__ == "__main__":
    main()

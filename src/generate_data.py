"""
Synthetic A/B test data generator — FreshCrate meal-kit price-framing experiment.

Scenario
--------
FreshCrate (a fictional meal-kit subscription) tests how price is shown on the
plan-selection page:
    control   -> "$79.92 / week"      (weekly box price)
    treatment -> "$9.99 / serving"    (price per serving)

Each row is one visitor who reached the plan-selection page during the 14-day
test. The funnel is:
    visit -> plan_selected -> checkout_started -> order_placed

Effects baked into the simulation (the analysis has to DISCOVER these):
    1. Treatment raises plan selection (the framing makes the price feel smaller).
    2. Treatment slightly lowers checkout completion ("sticker shock" when the
       weekly total appears at checkout).
    3. A novelty effect: the treatment lift is largest in the first few days
       and decays toward a steady state.
    4. Heterogeneous effects: bigger lift for new users and on mobile.
    5. Treatment nudges people toward 4-person plans (lower per-serving price).
    6. Realistic data-quality problems: duplicate logged rows, missing device,
       and a few tracking bugs (order placed with no checkout event).

Usage
-----
    python src/generate_data.py                      # defaults
    python src/generate_data.py --days 21 --daily-visitors 3000 --seed 7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Configuration: every assumption about the "true" world lives here
# --------------------------------------------------------------------------
@dataclass
class SimConfig:
    start_date: str = "2026-09-14"          # a Monday
    days: int = 14
    daily_visitors: int = 2800              # both groups combined, before weekday swing
    treatment_share: float = 0.50
    seed: int = 42

    # Share of traffic by segment
    p_mobile: float = 0.62
    p_new_user: float = 0.70

    # Baseline (control) funnel rates
    base_select: float = 0.22               # visit -> plan_selected
    base_checkout: float = 0.55             # plan_selected -> checkout_started
    base_order: float = 0.40                # checkout_started -> order_placed

    # Segment multipliers on plan selection
    select_mult_device: dict = field(default_factory=lambda: {"mobile": 0.88, "desktop": 1.20})
    select_mult_user: dict = field(default_factory=lambda: {"new": 0.90, "returning": 1.23})
    order_mult_device: dict = field(default_factory=lambda: {"mobile": 0.95, "desktop": 1.08})

    # Treatment effect on plan selection (relative lift, steady state)
    lift_new: float = 0.17
    lift_returning: float = 0.06
    lift_mobile_bonus: float = 0.04         # extra lift on mobile
    novelty_boost: float = 0.20             # extra lift on day 0 ...
    novelty_decay_days: float = 3.0         # ... decaying with this time constant

    # Treatment effect on checkout (sticker shock)
    treatment_checkout_mult: float = 0.92

    # Plans: (name, weekly price). Price per serving: 9.99 for 2-person, 8.99 for 4-person
    plans: tuple = (
        ("2p_3meals", 59.94),
        ("2p_4meals", 79.92),
        ("4p_3meals", 107.88),
        ("4p_4meals", 143.84),
    )
    plan_mix_control: tuple = (0.30, 0.40, 0.18, 0.12)
    plan_mix_treatment: tuple = (0.26, 0.38, 0.22, 0.14)
    shipping: float = 10.99
    promo_discounts: tuple = (0.0, 0.20, 0.40)   # first-box promo codes
    promo_probs: tuple = (0.55, 0.30, 0.15)

    # Data-quality noise
    dup_rate: float = 0.003                 # duplicated logging events
    missing_device_rate: float = 0.005
    tracking_bug_rate: float = 0.0008       # order logged without checkout event


# --------------------------------------------------------------------------
# 2. Traffic: who shows up, when, and which group they land in
# --------------------------------------------------------------------------
def simulate_traffic(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    # Meal-kit traffic peaks on Sunday/Monday (people plan the week's food)
    weekday_mult = {0: 1.15, 1: 1.00, 2: 0.95, 3: 0.92, 4: 0.85, 5: 0.90, 6: 1.25}
    dates = pd.date_range(cfg.start_date, periods=cfg.days, freq="D")

    frames = []
    for day_idx, d in enumerate(dates):
        n = rng.poisson(cfg.daily_visitors * weekday_mult[d.weekday()])
        seconds = rng.integers(0, 24 * 3600, size=n)
        frames.append(pd.DataFrame({
            "enrolled_at": d + pd.to_timedelta(seconds, unit="s"),
            "day_of_test": day_idx,
        }))
    df = pd.concat(frames, ignore_index=True).sort_values("enrolled_at").reset_index(drop=True)

    n = len(df)
    df.insert(0, "user_id", [f"u{100000 + i}" for i in rng.permutation(n)])
    df["group"] = np.where(rng.random(n) < cfg.treatment_share, "treatment", "control")
    df["device"] = np.where(rng.random(n) < cfg.p_mobile, "mobile", "desktop")
    df["user_type"] = np.where(rng.random(n) < cfg.p_new_user, "new", "returning")
    return df


# --------------------------------------------------------------------------
# 3. Behaviour: walk each visitor down the funnel
# --------------------------------------------------------------------------
def simulate_funnel(df: pd.DataFrame, cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    is_t = (df["group"] == "treatment").to_numpy()

    # Stage 1: plan selection — where the price framing is shown
    p_select = (
        cfg.base_select
        * df["device"].map(cfg.select_mult_device).to_numpy()
        * df["user_type"].map(cfg.select_mult_user).to_numpy()
    )
    lift = np.where(df["user_type"] == "new", cfg.lift_new, cfg.lift_returning)
    lift = lift + np.where(df["device"] == "mobile", cfg.lift_mobile_bonus, 0.0)
    novelty = cfg.novelty_boost * np.exp(-df["day_of_test"].to_numpy() / cfg.novelty_decay_days)
    p_select = np.where(is_t, p_select * (1 + lift + novelty), p_select)
    selected = rng.random(n) < np.clip(p_select, 0, 1)

    # Stage 2: checkout started — weekly total is revealed here
    p_checkout = np.where(is_t, cfg.base_checkout * cfg.treatment_checkout_mult, cfg.base_checkout)
    checkout = selected & (rng.random(n) < p_checkout)

    # Stage 3: order placed
    p_order = cfg.base_order * df["device"].map(cfg.order_mult_device).to_numpy()
    ordered = checkout & (rng.random(n) < p_order)

    df["plan_selected"] = selected.astype(int)
    df["checkout_started"] = checkout.astype(int)
    df["order_placed"] = ordered.astype(int)

    # Revenue for first box (zero for non-buyers)
    plan_names = np.array([p[0] for p in cfg.plans])
    plan_prices = np.array([p[1] for p in cfg.plans])
    idx_c = rng.choice(len(cfg.plans), size=n, p=cfg.plan_mix_control)
    idx_t = rng.choice(len(cfg.plans), size=n, p=cfg.plan_mix_treatment)
    plan_idx = np.where(is_t, idx_t, idx_c)
    discount = rng.choice(cfg.promo_discounts, size=n, p=cfg.promo_probs)
    revenue = plan_prices[plan_idx] * (1 - discount) + cfg.shipping

    df["plan_name"] = np.where(ordered, plan_names[plan_idx], None)
    df["revenue"] = np.where(ordered, revenue.round(2), 0.0)
    return df


# --------------------------------------------------------------------------
# 4. Mess: inject the kinds of problems real event data has
# --------------------------------------------------------------------------
def inject_data_issues(df: pd.DataFrame, cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # a) Missing device (user agent failed to parse)
    miss = rng.random(n) < cfg.missing_device_rate
    df.loc[miss, "device"] = np.nan

    # b) Tracking bug: order logged but checkout event dropped
    buyers = df.index[df["order_placed"] == 1]
    n_bug = max(1, int(len(df) * cfg.tracking_bug_rate))
    bug_rows = rng.choice(buyers, size=min(n_bug, len(buyers)), replace=False)
    df.loc[bug_rows, "checkout_started"] = 0

    # c) Duplicate events (client fired the exposure event twice)
    dup_rows = df.sample(frac=cfg.dup_rate, random_state=int(rng.integers(1e9)))
    df = pd.concat([df, dup_rows]).sort_values("enrolled_at").reset_index(drop=True)
    return df


def generate(cfg: SimConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    df = simulate_traffic(cfg, rng)
    df = simulate_funnel(df, cfg, rng)
    df = inject_data_issues(df, cfg, rng)
    df["date"] = df["enrolled_at"].dt.date
    cols = ["user_id", "enrolled_at", "date", "day_of_test", "group", "device", "user_type",
            "plan_selected", "checkout_started", "order_placed", "plan_name", "revenue"]
    return df[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FreshCrate A/B test data.")
    parser.add_argument("--days", type=int, default=SimConfig.days)
    parser.add_argument("--daily-visitors", type=int, default=SimConfig.daily_visitors)
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "ab_test_data.csv")
    args = parser.parse_args()

    cfg = SimConfig(days=args.days, daily_visitors=args.daily_visitors, seed=args.seed)
    df = generate(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Quick sanity printout
    print(f"Saved {len(df):,} rows -> {args.out}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}\n")
    print(df.head(8).to_string(index=False), "\n")
    summary = (df.groupby("group")
                 .agg(visitors=("user_id", "size"),
                      plan_selected=("plan_selected", "mean"),
                      checkout=("checkout_started", "mean"),
                      orders=("order_placed", "sum"),
                      conversion=("order_placed", "mean"),
                      revenue_per_visitor=("revenue", "mean")))
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()

# A/B Test Analysis Framework

**Did showing the price "per serving" instead of "per week" increase first orders, and how much can we trust the result?**

An end-to-end A/B test analysis of a price-framing experiment at a (fictional) meal-kit company. The project goes beyond a single p-value: it checks data quality and randomization, chooses the statistical test from the metric's type, measures power and effect size, catches a novelty effect and an underpowered design, breaks the result down by funnel stage and segment, and translates everything into a go/no-go recommendation for non-technical stakeholders. The analysis logic is packaged as a tested, reusable template for future experiments.

![Conversion funnel](reports/figures/07_conversion_funnel.png)

---

## Business question and scenario

**FreshCrate** sells weekly meal-kit subscriptions. On the plan-selection page, the team tested two ways of showing the price:

| | Control | Treatment |
|---|---|---|
| Price display | **"$79.92 / week"** | **"$9.99 / serving"** |

- **Test window:** 14 days (Sep 14 – Sep 27, 2026), 50/50 random split, 39,228 visitors after cleaning
- **Primary metric:** conversion, meaning the share of visitors who place a first order (binary)
- **Secondary metric:** first-box revenue per visitor (continuous)
- **Funnel:** visit → plan selected → checkout started → order placed
- **Pre-registered minimum detectable effect (MDE):** a 10% relative lift in conversion

The data is synthetic, produced by `src/generate_data.py`, with realistic effects and problems built in: a novelty effect, a checkout "sticker shock" tradeoff, effects that vary by segment, duplicate events, missing values, and tracking bugs. The analysis has to discover all of them without being told.

---

## Key findings

| Metric | Control | Treatment | Lift | 95% CI | p-value |
|---|---|---|---|---|---|
| Conversion (visit → order) | 4.87% | 5.70% | **+17.1%** | +0.39 to +1.28 pts | 0.0002 |
| Revenue per visitor | $4.23 | $5.14 | **+21.4%** | +$0.49 to +$1.32 | 0.00002 |
| Plan selection | 21.9% | 27.4% | +25.0% | +20.7% to +29.4% (relative) | < 0.00001 |

1. **The treatment works.** It adds about **8 first orders and $905 in first-box revenue per 1,000 visitors** (95% CI: $492 to $1,317). Welch's t-test, a bootstrap and Mann-Whitney all agree on revenue.
2. **But the test was underpowered for its own goal.** It had only **59% power** to detect the pre-registered 10% lift, and needed about 32,000 users per group (~23 days). A simulation shows significant results at this power overstate the true lift by about 1.3x, so +17% is probably optimistic.
3. **Novelty effect.** The plan-selection lift fell from **+32% in week 1 to +18% in week 2** (decay test p = 0.0002). Week-2 conversion lift was +12%, a better guide to the long-term effect.
4. **Funnel tradeoff.** The biggest impact is at plan selection (+25%). About **26% of that gain leaks at checkout**: the checkout rate is 4.9% lower (p = 0.019) when users first see the weekly total there.
5. **Segments.** All segments move in the same direction (+11% to +22%). No segment differs significantly from another (interaction p = 0.36 to 0.37), and every segment alone is underpowered (27–41% power).

**Recommendation: launch**, planning on the lower end of the range (~$0.5M a year in first-box revenue at current traffic). Next, test showing the weekly total next to the per-serving price, to recover the checkout leak. See the [executive summary](reports/executive_summary.md).

---

## Methodology

### How the test is chosen

The metric's type decides the test:

| Metric type | Test | Why |
|---|---|---|
| Binary (0/1), all 2×2 cells ≥ 10 | **Two-proportion z-test** | The difference of two proportions is approximately normal at this size. It is mathematically identical to the chi-square test on the 2×2 table (z² = χ², verified in the notebook). |
| Binary, any cell < 10 | **Fisher's exact test** | The normal approximation breaks down with few events. |
| Continuous, n > 25 × skewness² | **Welch's t-test** | The t-test needs the *mean* to be normal, not the raw data. The Central Limit Theorem delivers this at large n, even for revenue with 95% zeros (shown with a bootstrap). Welch's version doesn't assume equal variances. |
| Continuous with small n for its skew, or ordinal (ratings) | **Mann-Whitney U** | It compares ranks, so it is robust to outliers. Caveat: it does not test the difference in *means*. |

All tests are two-tailed at α = 0.05.

### What each statistic means

| Statistic | What it answers |
|---|---|
| **p-value** | If the change truly did nothing, how often would random assignment alone produce a difference this large? |
| **95% confidence interval** | The range of true effects consistent with the data: size and uncertainty together. |
| **Cohen's h** | Effect size for two proportions: 2·arcsin(√p₁) − 2·arcsin(√p₂). |
| **Cohen's d** | Effect size for two means: the difference divided by the pooled standard deviation. |
| **Power** | The probability of detecting a real effect of a given size (here, the 10% MDE). 80% is the usual target. |
| **Required sample size** | Users per group needed for 80% power at the MDE. It scales with 1 / effect², so half the effect needs about 4x the users. |
| **SRM check** | A chi-square test that the observed split matches the designed 50/50. A failure means assignment is broken. |
| **Holm correction** | Adjusts p-values when testing several segments or funnel steps, so the overall false-positive rate stays at 5%. |

### Analysis steps (in the notebook)

1. **EDA and data quality:** de-duplication, missing values, funnel-logic violations, SRM check, randomization balance, metric distributions.
2. **Test selection:** the decision function, tested on edge cases, plus the z = chi-square and CLT demonstrations.
3. **Statistical analysis:** hypotheses, p-values, CIs, effect sizes, power, required sample size, power curves, and a winner's-curse simulation.
4. **Novelty and segmentation:** daily lift, week-over-week comparison, a treatment × day logistic regression, and a per-segment analysis with power, Holm correction and interaction tests.
5. **Funnel:** reach per 1,000 visitors, step-by-step tests, and sizing of the checkout leak.

---

## Repository structure

```
ab-test-framework/
├── data/
│   ├── ab_test_data.csv          raw synthetic data (with injected data issues)
│   ├── ab_test_clean.csv         cleaned data written by the notebook
│   └── examples/
│       └── onboarding_test.csv   an unrelated test for trying the template
├── notebooks/
│   └── ab_test_analysis.ipynb    full analysis
├── reports/
│   ├── executive_summary.md      non-technical summary and recommendation
│   ├── template_output_freshcrate.csv
│   └── figures/                  all charts (PNG)
├── src/
│   ├── generate_data.py          synthetic data generator
│   └── ab_test_template.py       reusable A/B test template
├── tests/
│   └── test_ab_test_template.py  13 unit tests for the template
└── requirements.txt
```

---

## How to run

```bash
git clone https://github.com/ArpitaRaj27/ab-test-framework.git
cd ab-test-framework
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Generate the dataset (deterministic: seed 42)
python src/generate_data.py

# 2. Run the analysis notebook (interactive)
jupyter notebook notebooks/ab_test_analysis.ipynb
#    ...or execute it end-to-end from the command line
jupyter nbconvert --to notebook --execute --inplace notebooks/ab_test_analysis.ipynb

# 3. Run the template tests
pytest -q
```

To try a different scenario, change the assumptions in `SimConfig` at the top of `generate_data.py`, or pass options:

```bash
python src/generate_data.py --days 28 --daily-visitors 4000 --seed 7
```

The findings above are for the default settings (seed 42). Different settings produce different numbers; the methodology stays the same.

---

## Using the template on new data

`src/ab_test_template.py` works on any CSV with **one row per user**, a **group column** with exactly two values, and one or more **metric columns**.

### Command line

```bash
python src/ab_test_template.py data/examples/onboarding_test.csv \
    --group variant --control A --treatment B \
    --metrics activated csat days_to_activate \
    --ordinal csat \
    --mde 0.10 \
    --out reports/onboarding_results.csv
```

| Option | Meaning |
|---|---|
| `--group` | column holding the group label |
| `--control`, `--treatment` | the two group labels |
| `--metrics` | one or more metric columns |
| `--ordinal` | metrics on a ranked scale (ratings, NPS), which always use Mann-Whitney |
| `--mde` | minimum detectable effect as a relative lift (0.10 = 10%); enables power and sample-size output |
| `--alpha` | significance level (default 0.05) |
| `--split` | designed treatment share for the SRM check (default 0.5) |
| `--out` | save the summary table as CSV |

### Python

```python
import pandas as pd
import sys; sys.path.append("src")
from ab_test_template import analyze

df = pd.read_csv("data/examples/onboarding_test.csv")
table = analyze(df, group_col="variant", metrics=["activated", "csat", "days_to_activate"],
                control="A", treatment="B", mde=0.10, ordinal=["csat"])
```

### Output

For each metric, the template reports the **test chosen and why**, control and treatment values, the absolute and relative difference, a 95% CI, the p-value, an effect size, power at the MDE, the required sample size per group, and warnings:

- **SRM FAIL:** group sizes don't match the designed split
- **UNDERPOWERED:** below 80% power to detect the MDE
- **Small sample:** fewer than 100 users per group
- **Dropped rows:** missing metric values were removed

Example from the onboarding dataset:

```
-- activated  (binary)
   Test chosen:   Two-proportion z-test   because binary metric; smallest 2x2 cell = 115 >= 10
   Control:       28.75%     Treatment: 42.25%
   Difference:    +13.50 pts (+47.0% relative)   95% CI [+6.93, +20.07] pts
   p-value:       0.00007  -> SIGNIFICANT at alpha = 0.05
   Effect size:   Cohen's h = 0.2833
   Power at MDE:  14%   (80% needs 4,001 per group)
   ! UNDERPOWERED: 14% power for a 10% lift; need 4,001/group
```

---

## Limitations

- **Synthetic data.** The effects were planted, which lets us check that the methods recover them. Real data would add problems not simulated here, such as bots, cross-device users and seasonality.
- **First-box revenue only.** Long-term value (retention, repeat boxes) would need a longer follow-up window.
- **The novelty decay is modeled linearly** in the regression test. A longer test would allow fitting the decay curve directly.

---

**Author:** Arpita · [GitHub @ArpitaRaj27](https://github.com/ArpitaRaj27)

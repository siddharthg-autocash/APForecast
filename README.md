# APForecast Commander — Technical Documentation

## 1. System Overview

APForecast is a probabilistic **Accounts Payable (AP) cash forecasting system**.
Its objective is to estimate how much cash will leave the organization on a given day,
based on historical vendor payment behavior and currently outstanding checks.

Instead of using fixed due dates, APForecast models uncertainty using historical
clearance patterns and computes **expected cash outflow**.

**Core outputs**
- Expected cash outflow for a given day
- Vendor-level expected payments
- Forecast accuracy diagnostics via backtesting

---

## 2. Core Definitions

**Open Check**  
A check that has been issued (posted) but has not yet cleared the bank.

**Cleared Check**  
A check that has been confirmed as paid by the bank.

**Days to Settle**  
```
Days_to_Settle = Clear_Date − Post_Date
```

**Expected Cash**
```
Expected Cash = Amount × Probability of Clearing
```

---

## 3. Data Flow Architecture

```
Raw Historical Files
        ↓
Daily Outstanding / Cleared Files
        ↓
Ingestion & Reconciliation
        ↓
Master Ledger (Single Source of Truth)
        ↓
Forecast Engine
        ↓
Forecast / API / UI / Backtests
```

The **Master Ledger** is continuously updated and reused for all future runs.

---

## 4. Master Ledger Design

The Master Ledger tracks the full lifecycle of every check.

**Columns**
- Check_ID
- Vendor_ID
- Amount
- Post_Date
- Clear_Date
- Days_to_Settle
- Status (OPEN / CLEARED)

Once a check is cleared, its settlement data becomes training input for the model.

---

## 5. Probability Model (Mathematics)

Let:

```
D = random variable representing days taken to clear
```

### Empirical CDF

The model builds an empirical cumulative distribution function:

```
F(t) = P(D ≤ t)
```

This is computed directly from historical `Days_to_Settle` values.

---

### Conditional Probability (Core Forecast Logic)

We want to compute:

```
P(clear today | still open yesterday)
```

Using probability rules:

```
P(A | B) = P(A ∩ B) / P(B)
```

Applied here:

```
P(clear today | open yesterday)
= (F(today_age) − F(yesterday_age)) / (1 − F(yesterday_age))
```

This guarantees:
- No future data leakage
- Strict causality
- Proper conditioning on survival

---

## 6. Expected Cash Calculation

For each open check:

```
Expected Cash = Amount × Probability
```

Portfolio-level expected cash is the sum across all open checks.

This is **not a point prediction**, but a probability-weighted expectation.

---

## 7. Model Training Strategy

### Vendor-Specific Models
- Trained when vendor has ≥ 5 historical cleared checks
- Captures individual payment behavior

### Global Cohort Models (Fallback)
Used when vendor history is insufficient.

Cohorts are defined by check size:
- `< 10k` → Small
- `< 50k` → Medium
- `≥ 50k` → Large

Assumption: checks of similar size exhibit similar settlement behavior.

---

## 8. Forecast Engine Logic

For each open check:

1. Enforce causality (check must exist before forecast date)
2. Select model:
   - Vendor-specific → if available
   - Cohort-based → fallback
3. Compute current age and forecast age
4. Apply conditional probability formula
5. Multiply probability by amount

All probabilities are clamped to `[0, 1]`.

---

## 9. Backtesting Methodology

APForecast uses **walk-forward backtesting**, simulating real historical usage.

For each day in history:
- Train on data strictly before that day
- Predict cash for that day
- Compare against actual cleared cash

### Metrics

**Residual**
```
Residual = Actual − Predicted
```

**RMSE**
```
RMSE = sqrt(mean((Predicted − Actual)^2))
```

RMSE is also expressed as a percentage of mean actuals for interpretability.

Checks older than **45 days** are flagged as exceptions and excluded from prediction.

---

## 10. UI & Visualization

The Streamlit dashboard provides:

- Daily forecast execution
- Vendor payment profile (PDF + CDF curves)
- Delay scenario simulations
- Outstanding check landscape
- Excel exports
- Backtesting dashboards

Visualizations are **diagnostic**, not decorative.

---

## 11. API & Automation

FastAPI exposes:
```
GET /forecast/today?date=YYYY-MM-DD
```

Returns:
- Total outstanding
- Predicted cash for the day
- Vendor-level predictions

This enables integration with treasury systems and automation workflows.

---

## 12. Assumptions & Guarantees

### Assumptions
- Historical behavior predicts future behavior
- Vendors act independently
- Payment timing is probabilistic
- Very old checks are operational exceptions
- Cohort similarity holds

### Guarantees
- No look-ahead bias
- Fully explainable math
- Deterministic computation
- Reproducible forecasts
- Auditable backtests

---

## 13. Summary

APForecast is a transparent, probabilistic, and auditable cash forecasting engine.
It replaces brittle due-date logic with statistically grounded expectations,
providing finance teams with clarity, confidence, and control.

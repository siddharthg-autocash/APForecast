# src/apforecast/modeling/engine.py
import pandas as pd
import numpy as np
from src.apforecast.core.constants import *
from src.apforecast.modeling.probability import BayesianModel
from src.apforecast.modeling.cohorts import determine_cohort

class ForecastEngine:
    def __init__(self, ledger, overrides):
        self.ledger = ledger
        self.overrides = overrides
        self.models = self._train_models()

    def _train_models(self):
        models = {'SPECIFIC': {}, 'GLOBAL': {}}
        cleared = self.ledger[self.ledger[COL_STATUS] == STATUS_CLEARED]

        # 1. Train Specific Models
        vendor_counts = cleared[COL_VENDOR_ID].value_counts()
        valid_vendors = vendor_counts[vendor_counts >= 5].index
        for v_id in valid_vendors:
            data = cleared[cleared[COL_VENDOR_ID] == v_id][COL_DAYS_TO_SETTLE].values
            models['SPECIFIC'][v_id] = BayesianModel(data)

        # 2. Train Global Cohorts
        for cohort in [COHORT_SMALL, COHORT_MEDIUM, COHORT_LARGE]:
            mask = cleared[COL_AMOUNT].apply(determine_cohort) == cohort
            data = cleared[mask][COL_DAYS_TO_SETTLE].values
            models['GLOBAL'][cohort] = BayesianModel(data)
            
        return models
    
    def predict_check(self, check_row, forecast_date, current_date_override=None):
        """
        Returns a probability related to `forecast_date`.

        Behavior:
        - If `current_date_override` is provided:
            Return P(Clear on or before forecast_date | still unpaid at current_date_override).
            (This is a conditional cumulative probability. Marginal for a single day can be
            computed by differencing two calls with adjacent target dates.)
        - If `current_date_override` is NOT provided:
            Return unconditional P(Clear on or before forecast_date) (simple empirical CDF).
        """
        vendor_id = check_row[COL_VENDOR_ID]
        amount = check_row[COL_AMOUNT]
        post_date = pd.to_datetime(check_row[COL_POST_DATE])

        # compute ages (in days)
        forecast_age = (pd.to_datetime(forecast_date) - post_date).days

        if current_date_override is not None:
            current_age = (pd.to_datetime(current_date_override) - post_date).days
        else:
            current_age = None

        # 0. USER OVERRIDES (Top Priority)
        if vendor_id in self.overrides:
            rule = self.overrides[vendor_id]
            strategy = rule.get('Strategy')
            p1 = rule.get('Param_1')
            p2 = rule.get('Param_2')

            # For overrides, use forecast-based semantics:
            # e.g., FIXED_LAG = probability 1 when forecast_age >= p1
            if strategy == STRAT_HOLD:
                return 0.0
            if strategy == STRAT_PROB_OVERRIDE:
                return float(p1)
            if strategy == STRAT_FIXED_LAG:
                return 1.0 if forecast_age >= int(p1) else 0.0
            if strategy == STRAT_EXACT_DATE:
                return 1.0 if pd.to_datetime(forecast_date).normalize() == pd.to_datetime(p1).normalize() else 0.0
            if strategy == STRAT_WEEKDAY:
                target_day = str(p1).title()
                return float(p2) if pd.to_datetime(forecast_date).day_name() == target_day else 0.1

        # Helper to get a model for vendor / cohort
        def get_model_for(vendor, amt):
            if vendor in self.models['SPECIFIC']:
                return self.models['SPECIFIC'][vendor]
            cohort = determine_cohort(amt)
            return self.models['GLOBAL'].get(cohort)

        model = get_model_for(vendor_id, amount)

        # If no model at all, return 0
        if model is None or model.n == 0:
            return 0.0

        # CASE A: Conditional (we know state 'alive' at current_age)
        if current_age is not None:
            # If forecast is at-or-before the current age, no new probability
            if forecast_age <= current_age:
                return 0.0

            # If current_age beyond observed history -> can't compute conditional from specific model
            if current_age > model.max_observed_days:
                # allow fallback to GLOBAL if SPECIFIC unavailable; try global model
                # (get_model_for already tried GLOBAL for cohort when SPECIFIC missing)
                # but if this model is SPECIFIC and it's out-of-range, try global cohort model:
                cohort_model = self.models['GLOBAL'].get(determine_cohort(amount))
                if cohort_model and cohort_model.n > 0 and current_age <= cohort_model.max_observed_days:
                    model = cohort_model
                else:
                    return 0.0

            # Compute conditional cumulative:
            cdf_current = model.cdf(current_age)
            cdf_forecast = model.cdf(forecast_age)

            denom = 1.0 - cdf_current
            if denom <= 1e-12:
                return 0.0

            prob_conditional_cum = (cdf_forecast - cdf_current) / denom
            return max(0.0, min(1.0, prob_conditional_cum))

        # CASE B: Unconditional cumulative (no current_date_override provided)
        # just return empirical CDF at forecast_age
        return max(0.0, min(1.0, model.cdf(forecast_age)))

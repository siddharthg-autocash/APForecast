# src/apforecast/modeling/engine.py
import pandas as pd
import numpy as np
from src.apforecast.core.constants import *
from src.apforecast.modeling.probability import BayesianModel
from src.apforecast.modeling.cohorts import determine_cohort

class ForecastEngine:
    def __init__(self, ledger):
        self.ledger = ledger
        self.models = self._train_models()

    def _train_models(self):
        models = {'SPECIFIC': {}, 'GLOBAL': {}}
        cleared = self.ledger[self.ledger[COL_STATUS] == STATUS_CLEARED]

        # 1. Train Specific Models
        vendor_counts = cleared[COL_VENDOR_ID].value_counts()
        valid_vendors = vendor_counts[vendor_counts >= 5].index
        for v_id in valid_vendors:
            data = cleared[cleared[COL_VENDOR_ID] == v_id][COL_DAYS_TO_SETTLE].values
            data = np.array([int(x) for x in data if pd.notna(x)])
            models['SPECIFIC'][v_id] = BayesianModel(data)

        # 2. Train Global Cohorts
        for cohort in [COHORT_SMALL, COHORT_MEDIUM, COHORT_LARGE]:
            mask = cleared[COL_AMOUNT].apply(determine_cohort) == cohort
            data = cleared[mask][COL_DAYS_TO_SETTLE].values
            data = np.array([int(x) for x in data if pd.notna(x)])
            models['GLOBAL'][cohort] = BayesianModel(data)

        return models

    def predict_check(self, check_row, forecast_date, current_date_override=None):
        """
        Returns a probability for the check to clear on the effective forecast date.

        Key rules:
        - Ages are measured in business-days (business_days_between).
        - If forecast_date is a non-business day, the effective forecast becomes the next business day.
          This preserves probability mass (carry-forward) while not ageing over non-business days.
        - current_date_override will be normalized to the previous business day (if necessary).
        """
        vendor_id = check_row[COL_VENDOR_ID]
        amount = check_row[COL_AMOUNT]
        post_date = pd.to_datetime(check_row[COL_POST_DATE])
        forecast_date = pd.to_datetime(forecast_date)

        # Determine effective forecast date: freeze time over non-business days and carry forward
        if not is_business_day(forecast_date):
            effective_forecast_date = next_business_day(forecast_date)
        else:
            effective_forecast_date = forecast_date

        # Normalize/adjust current_date_override to previous business day (if provided)
        if current_date_override is not None:
            cdo = pd.to_datetime(current_date_override)
            if not is_business_day(cdo):
                current_date_override = prev_business_day(cdo)
            else:
                current_date_override = cdo

        # compute business-day ages
        forecast_age = business_days_between(post_date, effective_forecast_date)

        if current_date_override is not None:
            current_age = business_days_between(post_date, current_date_override)
        else:
            current_age = None

        # Helper to get a model for vendor / cohort
        def get_model_for(vendor, amt):
            if vendor in self.models['SPECIFIC']:
                return self.models['SPECIFIC'][vendor]
            cohort = determine_cohort(amt)
            return self.models['GLOBAL'].get(cohort)

        model = get_model_for(vendor_id, amount)

        if model is None or model.n == 0:
            return 0.0

        # CASE A: Conditional (we know state 'alive' at current_age)
        if current_age is not None:
            # If effective forecast is at-or-before the current age, no new probability
            if forecast_age <= current_age:
                return 0.0

            # If current_age beyond observed history -> try cohort fallback
            if current_age > model.max_observed_days:
                cohort_model = self.models['GLOBAL'].get(determine_cohort(amount))
                if cohort_model and cohort_model.n > 0 and current_age <= cohort_model.max_observed_days:
                    model = cohort_model
                else:
                    return 0.0

            cdf_current = model.cdf(current_age)
            cdf_forecast = model.cdf(forecast_age)

            denom = 1.0 - cdf_current
            if denom <= 1e-12:
                return 0.0

            prob_conditional_cum = (cdf_forecast - cdf_current) / denom
            return max(0.0, min(1.0, prob_conditional_cum))

        # CASE B: Unconditional cumulative
        return max(0.0, min(1.0, model.cdf(forecast_age)))

# ==========================================
# FILE: ./src/apforecast/modeling/engine.py
# ==========================================
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
        """
        Builds BayesianModels for:
        1. Every specific Vendor (if history >= 5)
        2. The 3 Global Cohorts
        """
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
            # Filter ledger by amount bucket
            # (In a real run, you'd add a 'Cohort' column to the ledger to speed this up)
            # Here we map on the fly for simplicity
            mask = cleared[COL_AMOUNT].apply(determine_cohort) == cohort
            data = cleared[mask][COL_DAYS_TO_SETTLE].values
            models['GLOBAL'][cohort] = BayesianModel(data)
            
        return models

    def predict_check(self, check_row, forecast_date, current_date_override=None):
        """
        Applies Logic Hierarchy: Override -> Specific -> Global
        
        current_date_override: Used to simulate "survival conditional on being alive until DATE".
                               If None, assumes check is being evaluated at forecast_date.
        """
        vendor_id = check_row[COL_VENDOR_ID]
        amount = check_row[COL_AMOUNT]
        post_date = check_row[COL_POST_DATE]
        
        # Calculate current age t
        # If we are simulating "Conditional Probability from Yesterday", we need
        # to know what "Yesterday" (current_date_override) was to calculate Age.
        if current_date_override:
            age = (current_date_override - post_date).days
        else:
            age = (forecast_date - post_date).days
        
        # CHECK 1: USER OVERRIDE
        if vendor_id in self.overrides:
            rule = self.overrides[vendor_id]
            strategy = rule['Strategy']
            p1 = rule['Param_1']
            p2 = rule['Param_2']

            if strategy == STRAT_HOLD:
                return 0.0
            
            if strategy == STRAT_PROB_OVERRIDE:
                return float(p1)

            if strategy == STRAT_FIXED_LAG:
                # If age is exactly lag, it clears. If age > lag, it's late (high prob).
                lag = int(p1)
                return 1.0 if age >= lag else 0.0

            if strategy == STRAT_EXACT_DATE:
                target_date = pd.to_datetime(p1)
                return 1.0 if forecast_date == target_date else 0.0

            if strategy == STRAT_WEEKDAY:
                target_day = str(p1).title() # e.g. "Friday"
                prob_weight = float(p2) if p2 else 0.9
                # If tomorrow is the target day?
                # Actually, we are forecasting if it clears TODAY (forecast_date).
                if forecast_date.day_name() == target_day:
                    return prob_weight
                else:
                    return 0.1 # Low chance on wrong day

        # CHECK 2: SPECIFIC HISTORY
        if vendor_id in self.models['SPECIFIC']:
            model = self.models['SPECIFIC'][vendor_id]
            return model.predict_survival_probability(age)

        # CHECK 3: GLOBAL COHORT
        cohort = determine_cohort(amount)
        model = self.models['GLOBAL'].get(cohort)
        if model:
            return model.predict_survival_probability(age)
        
        return 0.5 # Fallback if no data exists at all
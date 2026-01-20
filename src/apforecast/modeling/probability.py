# src/apforecast/modeling/probability.py
import numpy as np
import pandas as pd

class BayesianModel:
    def __init__(self, days_data):
        self.sorted_data = np.sort(days_data)
        self.n = len(days_data)
        
        # FIX: Store the maximum historical day observed.
        # This defines the "Settlement Max" for Step 1.
        self.max_observed_days = self.sorted_data[-1] if self.n > 0 else 0

    def cdf(self, t):
        """Empirical Cumulative Distribution Function"""
        if self.n == 0: return 0.0
        # searchsorted returns where t would fit in the list
        count = np.searchsorted(self.sorted_data, t, side='right')
        return count / self.n

    def predict_survival_probability(self, current_age, window=1):
        """
        Calculates P(Clear in Window | Alive at current_age).
        """
        # STEP 1: CHECK <= SETTLEMENT MAX
        # If the check is older than our entire history for this vendor, 
        # we return None. This signals the Engine to resort to Step 2 (Global).
        if self.n == 0 or current_age > self.max_observed_days:
            return None

        cdf_t = self.cdf(current_age)
        
        # EDGE CASE: If CDF is 1.0 (100% cleared), we can't divide by (1-1).
        # We return 0.0 because strictly speaking, locally, 
        # there is 0% probability remaining in *this* specific history.
        if cdf_t >= 0.9999:
            return 0.0

        cdf_t_window = self.cdf(current_age + window)
        
        # Standard conditional probability formula
        # P(A|B) = P(A and B) / P(B)
        # Probability of clearing next / Probability of surviving until now
        prob = (cdf_t_window - cdf_t) / (1.0 - cdf_t)
        
        return max(0.0, min(1.0, prob))
# src/apforecast/modeling/probability.py
import numpy as np
import pandas as pd

class BayesianModel:
    def __init__(self, days_data):
        """
        days_data: A list/array of integers representing 'Days_to_Settle' 
        from historical cleared checks.
        """
        self.sorted_data = np.sort(days_data)
        self.n = len(days_data)

    def cdf(self, t):
        """Empirical Cumulative Distribution Function"""
        if self.n == 0: return 0.0
        # Count how many items cleared <= t
        count = np.searchsorted(self.sorted_data, t, side='right')
        return count / self.n

    def predict_survival_probability(self, current_age, window=1):
        """
        Formula: P = (CDF(t + window) - CDF(t)) / (1 - CDF(t))
        """
        cdf_t = self.cdf(current_age)
        cdf_t_window = self.cdf(current_age + window)
        
        # Handle edge case where cdf_t is 1.0 (already passed max history)
        if cdf_t >= 1.0:
            return 0.05 # Decay probability for extreme outliers
            
        prob = (cdf_t_window - cdf_t) / (1.0 - cdf_t)
        return max(0.0, min(1.0, prob))
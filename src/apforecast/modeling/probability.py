# src/apforecast/modeling/probability.py

import numpy as np


# -----------------------------
# ECDF Builder
# -----------------------------

def build_ecdf(days_to_settle: list[int]):
    """
    Build an empirical CDF function from historical settlement days.
    Returns a callable CDF(t).
    """

    values = np.sort(np.array(days_to_settle))

    def cdf(t: int) -> float:
        if t < 0:
            return 0.0
        return np.searchsorted(values, t, side="right") / len(values)

    return cdf


# -----------------------------
# Bayesian Survival Probability
# -----------------------------

def conditional_clear_probability(
    cdf_fn,
    current_age: int,
    window: int = 1,
) -> float:
    """
    P(clear in [t, t+window] | survived to t)
    """

    cdf_t = cdf_fn(current_age)
    cdf_tw = cdf_fn(current_age + window)

    survival_prob = 1.0 - cdf_t

    # already extremely unlikely to survive
    if survival_prob <= 0:
        return 0.0

    prob = (cdf_tw - cdf_t) / survival_prob

    # numerical safety
    return max(0.0, min(1.0, prob))

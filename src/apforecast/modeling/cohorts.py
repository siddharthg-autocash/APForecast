# src/apforecast/modeling/cohorts.py

from apforecast.core.constants import COHORT_THRESHOLDS


def assign_cohort(amount: float) -> str:
    """
    Assign check to a global cohort based on amount.
    """

    if amount <= COHORT_THRESHOLDS["SMALL_MAX"]:
        return "STABLE"

    if amount <= COHORT_THRESHOLDS["MED_MAX"]:
        return "VOLATILE"

    return "LAZY"

# src/apforecast/modeling/cohorts.py
from src.apforecast.core.constants import *

def determine_cohort(amount):
    if amount < THRESHOLD_SMALL:
        return COHORT_SMALL
    elif amount < THRESHOLD_LARGE:
        return COHORT_MEDIUM
    else:
        return COHORT_LARGE
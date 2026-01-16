# src/apforecast/core/dates.py

from datetime import date


def parse_run_date(date_str: str) -> date:
    """
    Parse CLI date argument: '17-01-2026' -> datetime.date
    """
    return date.fromisoformat(
        "-".join(reversed(date_str.split("-")))
    )


def age_in_days(post_date: date, run_date: date) -> int:
    """
    Age of a check in days as of run_date.
    """
    return (run_date - post_date).days


def days_between(start: date, end: date) -> int:
    """
    Inclusive difference helper.
    """
    return (end - start).days

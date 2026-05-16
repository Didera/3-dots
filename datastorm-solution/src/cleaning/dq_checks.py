"""
Reusable, parameterizable data quality checks for the DataStorm 7.0 pipeline.

Every check function follows the same signature:
    check_*(df, ...) -> (clean_df, rejected_df)

Rejected rows are tagged with:
  - `rejection_reason`: Human-readable description of the failure.
  - `check_name`:        Machine-readable check identifier.

These checks are designed to be composed: run multiple checks sequentially,
accumulating rejected records into a single rejected store.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

import pandas as pd


# ---------------------------------------------------------------------------
# Helper to tag & split
# ---------------------------------------------------------------------------

def _split(
    df: pd.DataFrame,
    mask: pd.Series,
    reason: str,
    check_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into (clean, rejected) using a boolean *mask*.

    Rows where *mask* is ``True`` are **rejected**.
    """
    rejected = df[mask].copy()
    if not rejected.empty:
        rejected["rejection_reason"] = reason
        rejected["check_name"] = check_name
    clean = df[~mask].copy()
    return clean, rejected


# ---------------------------------------------------------------------------
# 1. Duplicate check
# ---------------------------------------------------------------------------

def check_duplicates(
    df: pd.DataFrame,
    key_cols: List[str],
    keep: str = "first",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect and remove duplicate records based on *key_cols*.

    Parameters
    ----------
    df : DataFrame
    key_cols : list[str]  — columns forming the composite primary key.
    keep : str            — which duplicate to keep (``'first'`` | ``'last'``).
    """
    dup_mask = df.duplicated(subset=key_cols, keep=keep)
    return _split(
        df, dup_mask,
        reason=f"Duplicate on key {key_cols}",
        check_name="duplicate_check",
    )


# ---------------------------------------------------------------------------
# 2. Null / missing value check
# ---------------------------------------------------------------------------

def check_nulls(
    df: pd.DataFrame,
    mandatory_cols: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag records where any of *mandatory_cols* contains null / NaN."""
    null_mask = df[mandatory_cols].isnull().any(axis=1)
    return _split(
        df, null_mask,
        reason=f"Null value in mandatory columns {mandatory_cols}",
        check_name="null_check",
    )


# ---------------------------------------------------------------------------
# 3. Referential integrity check
# ---------------------------------------------------------------------------

def check_referential_integrity(
    df: pd.DataFrame,
    col: str,
    ref_df: pd.DataFrame,
    ref_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate that every value in *df[col]* exists in *ref_df[ref_col]*."""
    valid_keys: Set = set(ref_df[ref_col].dropna().unique())
    bad_mask = ~df[col].isin(valid_keys)
    return _split(
        df, bad_mask,
        reason=f"Referential integrity failure: {col} not in {ref_col}",
        check_name="referential_integrity_check",
    )


# ---------------------------------------------------------------------------
# 4. Value range check
# ---------------------------------------------------------------------------

def check_value_range(
    df: pd.DataFrame,
    col: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assert that *df[col]* falls within [min_val, max_val]."""
    mask = pd.Series(False, index=df.index)
    if min_val is not None:
        mask = mask | (df[col] < min_val)
    if max_val is not None:
        mask = mask | (df[col] > max_val)
    return _split(
        df, mask,
        reason=f"Value out of range for '{col}': expected [{min_val}, {max_val}]",
        check_name="value_range_check",
    )


# ---------------------------------------------------------------------------
# 5. Format / pattern check
# ---------------------------------------------------------------------------

def check_format(
    df: pd.DataFrame,
    col: str,
    pattern: str,
    description: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate that *df[col]* matches a regex *pattern*."""
    regex = re.compile(pattern)
    bad_mask = ~df[col].astype(str).apply(lambda v: bool(regex.match(v)))
    desc = description or f"pattern '{pattern}'"
    return _split(
        df, bad_mask,
        reason=f"Format check failed for '{col}': expected {desc}",
        check_name="format_check",
    )


# ---------------------------------------------------------------------------
# 6. Valid categories check
# ---------------------------------------------------------------------------

def check_valid_categories(
    df: pd.DataFrame,
    col: str,
    valid_set: Set[str],
    case_sensitive: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate that *df[col]* contains only values from *valid_set*."""
    if case_sensitive:
        bad_mask = ~df[col].isin(valid_set)
    else:
        lower_valid = {v.lower() for v in valid_set}
        bad_mask = ~df[col].astype(str).str.lower().isin(lower_valid)
    return _split(
        df, bad_mask,
        reason=f"Invalid category in '{col}': expected one of {valid_set}",
        check_name="category_check",
    )


# ---------------------------------------------------------------------------
# 7. Composite runner
# ---------------------------------------------------------------------------

def run_checks(
    df: pd.DataFrame,
    checks: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a sequence of check functions, accumulating rejected records.

    Parameters
    ----------
    df : DataFrame          — the input data.
    checks : list of tuples — each tuple is ``(check_fn, kwargs_dict)``.

    Returns
    -------
    (clean_df, all_rejected_df)
    """
    all_rejected = []
    current = df.copy()

    for check_fn, kwargs in checks:
        current, rejected = check_fn(current, **kwargs)
        if not rejected.empty:
            all_rejected.append(rejected)

    if all_rejected:
        rejected_df = pd.concat(all_rejected, ignore_index=True)
    else:
        rejected_df = pd.DataFrame()

    return current, rejected_df

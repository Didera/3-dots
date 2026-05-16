"""Generate a comprehensive data cleaning report (Markdown).

Reads the summary dicts produced by each cleaning step and generates
a structured report documenting:
  - Per-dataset row counts, pass/fail rates, and rejection breakdowns.
  - Anomalies discovered (typos, swaps, negative volumes, outliers).
  - Rejected records file paths for audit.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SILVER = _PROJECT_ROOT / "data" / "silver"
REPORT_PATH = _SILVER / "cleaning_report.md"


def generate_cleaning_report(summaries: List[dict]) -> str:
    """Generate the cleaning report markdown and write to disk.

    Parameters
    ----------
    summaries : list[dict]
        Each dict is the return value of a ``clean_*()`` function.

    Returns
    -------
    str — the markdown content.
    """
    lines = []
    lines.append("# Data Cleaning Report")
    lines.append(f"")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Pipeline:** DataStorm 7.0 -- Bronze -> Silver")
    lines.append("")

    # --- Overall summary table ---
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Dataset | Initial Rows | Clean Rows | Rejected Rows | Pass Rate |")
    lines.append("|---------|-------------|-----------|--------------|-----------|")

    total_initial = 0
    total_clean = 0
    total_rejected = 0

    for s in summaries:
        init_r = s.get("initial_rows", 0)
        clean_r = s.get("clean_rows", 0)
        rej_r = s.get("rejected_rows", 0)
        total_initial += init_r
        total_clean += clean_r
        total_rejected += rej_r
        rate = f"{clean_r / init_r * 100:.2f}%" if init_r else "N/A"
        lines.append(
            f"| {s['dataset']} | {init_r:,} | {clean_r:,} | {rej_r:,} | {rate} |"
        )

    total_rate = (
        f"{total_clean / total_initial * 100:.2f}%" if total_initial else "N/A"
    )
    lines.append(
        f"| **TOTAL** | **{total_initial:,}** | **{total_clean:,}** "
        f"| **{total_rejected:,}** | **{total_rate}** |"
    )
    lines.append("")

    # --- Per-dataset details ---
    lines.append("---")
    lines.append("")
    lines.append("## Per-Dataset Details")
    lines.append("")

    for s in summaries:
        lines.append(f"### {s['dataset']}")
        lines.append("")
        lines.append(f"- **Initial rows:** {s.get('initial_rows', 0):,}")
        lines.append(f"- **Clean rows:** {s.get('clean_rows', 0):,}")
        lines.append(f"- **Rejected rows:** {s.get('rejected_rows', 0):,}")
        lines.append(f"- **Clean file:** `{s.get('clean_path', 'N/A')}`")
        lines.append(f"- **Rejected file:** `{s.get('rejected_path', 'N/A')}`")
        lines.append("")

        # Rejection breakdown
        breakdown = s.get("rejection_breakdown", {})
        if breakdown:
            lines.append("**Rejection Breakdown:**")
            lines.append("")
            lines.append("| Check | Rejected Count |")
            lines.append("|-------|---------------|")
            for check, count in sorted(breakdown.items()):
                lines.append(f"| {check} | {count:,} |")
            lines.append("")

        # Special annotations per dataset
        if "typos_fixed_type" in s:
            lines.append("**Data Quality Corrections:**")
            lines.append(f"- Outlet_Type typos auto-corrected: {s['typos_fixed_type']}")
            lines.append(f"- Outlet_Size typos auto-corrected: {s['typos_fixed_size']}")
            lines.append(
                f"- Null Outlet_Size imputed as 'Unknown': {s['null_size_imputed']}"
            )
            lines.append(
                f"- Final Outlet_Type values: {s.get('final_outlet_types', [])}"
            )
            lines.append(
                f"- Final Outlet_Size values: {s.get('final_outlet_sizes', [])}"
            )
            lines.append("")

        if "swaps_fixed" in s:
            lines.append("**Data Quality Corrections:**")
            lines.append(
                f"- Swapped latitude/longitude auto-corrected: {s['swaps_fixed']}"
            )
            lines.append("")

        if "returns_flagged" in s:
            lines.append("**Anomaly Flags (kept in clean data):**")
            lines.append(
                f"- Negative volume rows flagged as returns: {s['returns_flagged']:,}"
            )
            lines.append(
                f"- Extreme volume outliers flagged (3x IQR): {s['outliers_flagged']:,}"
            )
            lines.append("")

        if "bad_dates_found" in s:
            lines.append("**Data Quality Notes:**")
            lines.append(f"- Unparseable dates found: {s['bad_dates_found']}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # --- DQ Checks Applied ---
    lines.append("## Reusable DQ Checks Applied")
    lines.append("")
    lines.append("| Check | Description |")
    lines.append("|-------|-------------|")
    lines.append("| `duplicate_check` | Detect duplicate records on configurable key columns |")
    lines.append("| `null_check` | Flag records with null/NaN in mandatory columns |")
    lines.append("| `referential_integrity_check` | Validate FK values exist in reference dataset |")
    lines.append("| `value_range_check` | Assert numeric fields within [min, max] bounds |")
    lines.append("| `format_check` | Validate field format via regex pattern |")
    lines.append("| `category_check` | Validate categorical values against allowed set |")
    lines.append("")
    lines.append("> All checks are parameterizable and applied consistently across datasets.")
    lines.append("> Rejected records are quarantined with `rejection_reason` and `check_name`")
    lines.append("> columns — never silently dropped.")
    lines.append("")

    report = "\n".join(lines)

    _SILVER.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  Cleaning report saved: {REPORT_PATH}")

    return report

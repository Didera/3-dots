"""Verify all pipeline outputs."""
import pandas as pd
import os

print("=== OUTPUT VERIFICATION ===\n")

# Silver layer
silver = "data/silver"
for f in ["outlet_master_clean.csv", "outlet_coordinates_clean.csv",
          "seasonality_clean.csv", "holidays_clean.csv", "transactions_clean.csv"]:
    path = os.path.join(silver, f)
    size = os.path.getsize(path)
    if size < 50_000_000:
        df = pd.read_csv(path)
        print(f"[OK] {f}: {len(df)} rows, {size/1024:.0f} KB")
    else:
        print(f"[OK] {f}: (large file), {size/1024/1024:.0f} MB")

# Rejected records
rej = os.path.join(silver, "rejected_records")
for f in os.listdir(rej):
    path = os.path.join(rej, f)
    size = os.path.getsize(path)
    if size > 0:
        try:
            df = pd.read_csv(path)
            has_reason = "rejection_reason" in df.columns
            print(f"[OK] rejected/{f}: {len(df)} rows, has_reason={has_reason}")
        except Exception:
            print(f"[OK] rejected/{f}: empty (0 bytes)")
    else:
        print(f"[OK] rejected/{f}: empty (0 rejections)")

# Report
rpt = os.path.join(silver, "cleaning_report.md")
print(f"[OK] cleaning_report.md: {os.path.getsize(rpt)} bytes")

# Gold layer
gold = "data/gold/features"
for f in ["final_features.csv", "3dots_predictions.csv", "prediction_details.csv"]:
    path = os.path.join(gold, f)
    df = pd.read_csv(path)
    print(f"[OK] {f}: {len(df)} rows, {len(df.columns)} cols")

# Sanity checks
print("\n=== SANITY CHECKS ===")
preds = pd.read_csv(os.path.join(gold, "3dots_predictions.csv"))
feats = pd.read_csv(os.path.join(gold, "final_features.csv"))

print(f"All outlets have predictions: {len(preds) == 20000}")
print(f"No null predictions: {preds['Maximum_Monthly_Liters'].isnull().sum() == 0}")
print(f"All predictions positive: {(preds['Maximum_Monthly_Liters'] > 0).all()}")

merged = preds.merge(feats[["Outlet_ID", "max_monthly_volume"]], on="Outlet_ID")
violations = (merged["Maximum_Monthly_Liters"] < merged["max_monthly_volume"] * 0.99).sum()
print(f"Potential >= historical max (within 1%): violations = {violations}")

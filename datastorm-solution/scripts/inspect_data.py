"""Quick data inspection script."""
import pandas as pd
import os

base = "data/bronze/raw"

# Transactions summary
fsize = os.path.getsize(os.path.join(base, "transactions_history_final.csv"))
print(f"Transactions file size: {fsize / 1024 / 1024:.1f} MB")

df = pd.read_csv(os.path.join(base, "transactions_history_final.csv"))
print(f"Transactions rows: {len(df):,}")
print(f"Unique Outlet_IDs: {df['Outlet_ID'].nunique()}")
print(f"Unique SKUs: {df['SKU_ID'].nunique()}")
print(f"Unique Distributors: {df['Distributor_ID'].nunique()}")
print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
print(f"Month range: {df['Month'].min()} - {df['Month'].max()}")
print(f"\nNull counts:")
print(df.isnull().sum())
print(f"\nDescribe:")
print(df.describe())

# Outlet master unique values
om = pd.read_csv(os.path.join(base, "outlet_master.csv"))
print(f"\n=== OUTLET MASTER UNIQUE VALUES ===")
print(f"Outlet_Size: {om['Outlet_Size'].unique()}")
print(f"Outlet_Type: {om['Outlet_Type'].unique()}")
print(f"Cooler_Count range: {om['Cooler_Count'].min()} - {om['Cooler_Count'].max()}")

# Holiday date range
hol = pd.read_csv(os.path.join(base, "holiday_list.csv"))
print(f"\n=== HOLIDAYS ===")
print(f"Holiday_Type unique: {hol['Holiday_Type'].unique()}")
print(f"Date range: {hol['Date'].min()} - {hol['Date'].max()}")

# Distributor seasonality
ds = pd.read_csv(os.path.join(base, "distributor_seasonality_details.csv"))
print(f"\n=== DISTRIBUTOR SEASONALITY ===")
print(f"Unique distributors: {ds['Distributor_ID'].nunique()}")
print(f"Seasonality_Index: {ds['Seasonality_Index'].unique()}")

# Check for duplicate outlets
print(f"\n=== DUPLICATES ===")
print(f"Outlet master duplicates: {om.duplicated(subset='Outlet_ID').sum()}")
oc = pd.read_csv(os.path.join(base, "outlet_coordinates.csv"))
print(f"Outlet coordinates duplicates: {oc.duplicated(subset='Outlet_ID').sum()}")

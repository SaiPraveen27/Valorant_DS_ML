import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("./valorant_dataset_v3.csv")

print("Original shape:", df.shape)

# 2. Check missing values before cleaning
print("\nMissing values before cleaning:")
print(df.isna().sum())

# 3. Convert comma-separated numeric text columns to numeric
comma_columns = ["assists", "damage_received", "headshots", "traded", "kills", "matches", "deaths", "damage"]
print("\n", comma_columns)

for col in comma_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Check missing values after conversion
print("\nMissing values after numeric conversion:")
print(df.isna().sum())

# 5. Remove rows with missing values
df = df.dropna()

# 6. Remove duplicate player entries, if any
print("\nDuplicate player rows:", df["player"].duplicated().sum())
df = df.drop_duplicates(subset="player", keep="first")

# 7. Remove invalid rows
# A player must have played at least one match
df = df[df["matches"] > 0]

# Deaths can be zero, but kills/damage should not be negative
numeric_check_columns = [
    "assists",
    "damage_received",
    "headshots",
    "traded",
    "kills",
    "matches",
    "deaths",
    "damage"
]

for col in numeric_check_columns:
    df = df[df[col] >= 0]

# 8. Reset index after removing rows
df = df.reset_index(drop=True)

# 9. Verify final data types
print("\nFinal shape:", df.shape)
print("\nFinal data types:")
print(df.dtypes)

# 10. Final missing-value check
print("\nMissing values after cleaning:")
print(df.isna().sum())

# 11. Preview cleaned dataset
df.head()

#Save the Changes into the Dataset
df.to_csv("./valorant_dataset_cleaned.csv", index=False)


import os
import pandas as pd

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    r"C:\Coding Section\IIITV\Major Projects"
    r"\Solar Panel Power Generator Prediction"
)

DATASET_FOLDER = os.path.join(PROJECT_ROOT, "dataset")

OLD_FILE = os.path.join(
    DATASET_FOLDER,
    "nasa_power_india_daily.csv"
)

RH_FILE = os.path.join(
    DATASET_FOLDER,
    "nasa_power_india_RH2M_daily.csv"
)

OUTPUT_FILE = os.path.join(
    DATASET_FOLDER,
    "nasa_power_india_combined_daily.csv"
)


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

print("Project folder:", PROJECT_ROOT)
print("Dataset folder:", DATASET_FOLDER)

if not os.path.isfile(OLD_FILE):
    raise FileNotFoundError(f"Original dataset not found: {OLD_FILE}")

if not os.path.isfile(RH_FILE):
    raise FileNotFoundError(f"RH2M dataset not found: {RH_FILE}")


# ============================================================
# 3. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

df_original = pd.read_csv(OLD_FILE)
df_rh = pd.read_csv(RH_FILE)

print("Original dataset shape:", df_original.shape)
print("RH2M dataset shape:", df_rh.shape)


# ============================================================
# 4. STANDARDIZE COLUMN NAMES
# ============================================================

df_original.columns = df_original.columns.str.strip().str.lower()
df_rh.columns = df_rh.columns.str.strip().str.lower()

# Support either RH2M or rh2m column name
if "rh2m" not in df_rh.columns:
    if "rh2m" in df_rh.columns:
        pass
    elif "RH2M" in df_rh.columns:
        df_rh.rename(columns={"RH2M": "rh2m"}, inplace=True)
    else:
        raise ValueError("RH2M column not found in RH dataset.")

# Check required columns
required_original = [
    "state", "lat", "lon", "date",
    "tmax", "tmin", "wind_speed", "solar_radiation"
]

required_rh = ["state", "date", "rh2m"]

for col in required_original:
    if col not in df_original.columns:
        raise ValueError(f"Missing column in original dataset: {col}")

for col in required_rh:
    if col not in df_rh.columns:
        raise ValueError(f"Missing column in RH2M dataset: {col}")


# ============================================================
# 5. NORMALIZE JOIN KEYS
# ============================================================

def normalize_state(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )


df_original["_state_key"] = normalize_state(df_original["state"])
df_rh["_state_key"] = normalize_state(df_rh["state"])


# ============================================================
# 5. NORMALIZE JOIN KEYS
# ============================================================

def normalize_state(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )


df_original["_state_key"] = normalize_state(df_original["state"])
df_rh["_state_key"] = normalize_state(df_rh["state"])

# Parse dates, supporting mixed date formats
df_original["_date_key"] = pd.to_datetime(
    df_original["date"].astype(str).str.strip(),
    format="mixed",
    dayfirst=True,
    errors="coerce"
).dt.normalize()

df_rh["_date_key"] = pd.to_datetime(
    df_rh["date"].astype(str).str.strip(),
    format="mixed",
    dayfirst=True,
    errors="coerce"
).dt.normalize()

# Identify invalid dates
invalid_original = df_original["_date_key"].isna()
invalid_rh = df_rh["_date_key"].isna()

if invalid_original.any() or invalid_rh.any():

    print("\nInvalid date values detected!")

    if invalid_original.any():
        print("\nOriginal dataset invalid dates:")
        print(
            df_original.loc[invalid_original, "date"]
            .head(20)
            .to_list()
        )
        print("Total invalid original dates:", invalid_original.sum())

    if invalid_rh.any():
        print("\nRH2M dataset invalid dates:")
        print(
            df_rh.loc[invalid_rh, "date"]
            .head(20)
            .to_list()
        )
        print("Total invalid RH2M dates:", invalid_rh.sum())

    raise ValueError(
        "Please inspect the invalid date values printed above."
    )

print("\nDate normalization successful!")


if df_original["_date_key"].isna().any():
    raise ValueError("Invalid dates found in original dataset.")

if df_rh["_date_key"].isna().any():
    raise ValueError("Invalid dates found in RH2M dataset.")


# ============================================================
# 6. CHECK DUPLICATE KEYS IN RH2M DATA
# ============================================================

duplicate_count = df_rh.duplicated(
    subset=["_state_key", "_date_key"]
).sum()

if duplicate_count > 0:
    duplicates = df_rh[
        df_rh.duplicated(
            subset=["_state_key", "_date_key"],
            keep=False
        )
    ]

    print("\nDuplicate RH2M keys found:")
    print(duplicates.head(10))

    raise ValueError(
        f"RH2M dataset contains {duplicate_count} duplicate state/date keys. "
        "Resolve these before merging."
    )


# ============================================================
# 7. MERGE DATASETS
# ============================================================

print("\nMerging datasets...")

df_final = pd.merge(
    df_original,
    df_rh[["_state_key", "_date_key", "rh2m"]],
    on=["_state_key", "_date_key"],
    how="left",
    validate="many_to_one",
    indicator=True
)


# ============================================================
# 8. CHECK MERGE RESULTS
# ============================================================

matched = (df_final["_merge"] == "both").sum()
unmatched = (df_final["_merge"] == "left_only").sum()

print("\n========== MERGE REPORT ==========")
print("Original rows:", len(df_original))
print("RH2M rows:", len(df_rh))
print("Final rows:", len(df_final))
print("Matched rows:", matched)
print("Unmatched rows:", unmatched)

if len(df_final) != len(df_original):
    raise ValueError(
        "Row count changed unexpectedly. Check the merge keys."
    )

if unmatched > 0:
    print("\nWARNING: Some rows did not find matching RH2M values.")
    print("Sample unmatched records:")

    print(
        df_final.loc[
            df_final["_merge"] == "left_only",
            ["state", "date"]
        ].head(15)
    )


# ============================================================
# 9. CLEAN FINAL DATASET
# ============================================================

df_final.drop(
    columns=["_state_key", "_date_key", "_merge"],
    inplace=True
)

# Arrange columns in the required order
final_columns = [
    "state",
    "lat",
    "lon",
    "date",
    "tmax",
    "tmin",
    "wind_speed",
    "solar_radiation",
    "rh2m"
]

df_final = df_final[final_columns]


# ============================================================
# 10. SAVE FINAL DATASET
# ============================================================

df_final.to_csv(OUTPUT_FILE, index=False)

print("\n========== FINAL DATASET ==========")
print("Saved successfully!")
print("Output file:", OUTPUT_FILE)
print("Final shape:", df_final.shape)
print("\nColumns:")
print(df_final.columns.tolist())

print("\nMissing values:")
print(df_final.isnull().sum())

print("\nFirst five rows:")
print(df_final.head())

print("\nMerge process completed successfully!")

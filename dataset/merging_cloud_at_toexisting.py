
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
    "nasa_power_india_combined_daily.csv"
)

CLOUD_PRESSURE_FILE = os.path.join(
    DATASET_FOLDER,
    "nasa_power_india_cloud_pressure_daily.csv"
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
    raise FileNotFoundError(
        f"Existing dataset not found: {OLD_FILE}"
    )

if not os.path.isfile(CLOUD_PRESSURE_FILE):
    raise FileNotFoundError(
        f"Cloud-pressure dataset not found: {CLOUD_PRESSURE_FILE}"
    )


# ============================================================
# 3. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

df_original = pd.read_csv(OLD_FILE)
df_new = pd.read_csv(CLOUD_PRESSURE_FILE)

print("Existing dataset shape:", df_original.shape)
print("New dataset shape:", df_new.shape)


# ============================================================
# 4. STANDARDIZE COLUMN NAMES
# ============================================================

df_original.columns = (
    df_original.columns.str.strip().str.lower()
)

df_new.columns = (
    df_new.columns.str.strip().str.lower()
)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_original = [
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

required_new = [
    "state",
    "date",
    "cloud_cover",
    "atmospheric_pressure"
]

for col in required_original:
    if col not in df_original.columns:
        raise ValueError(
            f"Missing column in existing dataset: {col}"
        )

for col in required_new:
    if col not in df_new.columns:
        raise ValueError(
            f"Missing column in new dataset: {col}"
        )


# ============================================================
# 6. NORMALIZE STATE NAMES
# ============================================================

def normalize_state(series):

    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )


df_original["_state_key"] = normalize_state(
    df_original["state"]
)

df_new["_state_key"] = normalize_state(
    df_new["state"]
)


# ============================================================
# 7. NORMALIZE DATES
# ============================================================

def normalize_dates(series):

    # Supports DD-MM-YYYY and other common date formats
    return pd.to_datetime(
        series.astype(str).str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce"
    ).dt.normalize()


df_original["_date_key"] = normalize_dates(
    df_original["date"]
)

df_new["_date_key"] = normalize_dates(
    df_new["date"]
)


# Check invalid dates

invalid_original = df_original["_date_key"].isna()
invalid_new = df_new["_date_key"].isna()

if invalid_original.any() or invalid_new.any():

    print("\nInvalid date values detected!")

    if invalid_original.any():
        print("\nInvalid dates in existing dataset:")
        print(
            df_original.loc[
                invalid_original, "date"
            ].head(20).to_list()
        )

    if invalid_new.any():
        print("\nInvalid dates in new dataset:")
        print(
            df_new.loc[
                invalid_new, "date"
            ].head(20).to_list()
        )

    raise ValueError(
        "Please correct invalid dates before merging."
    )

print("\nState and date normalization successful!")


# ============================================================
# 8. CHECK DUPLICATE KEYS IN NEW DATASET
# ============================================================

duplicate_count = df_new.duplicated(
    subset=["_state_key", "_date_key"]
).sum()

if duplicate_count > 0:

    duplicates = df_new[
        df_new.duplicated(
            subset=["_state_key", "_date_key"],
            keep=False
        )
    ]

    print("\nDuplicate keys found in new dataset:")
    print(
        duplicates[
            ["state", "date"]
        ].head(20)
    )

    raise ValueError(
        f"New dataset contains {duplicate_count} duplicate "
        "state/date keys. Resolve them before merging."
    )


# ============================================================
# 9. MERGE DATASETS
# ============================================================

print("\nMerging datasets using state and date...")

df_final = pd.merge(
    df_original,
    df_new[
        [
            "_state_key",
            "_date_key",
            "cloud_cover",
            "atmospheric_pressure"
        ]
    ],
    on=["_state_key", "_date_key"],
    how="left",
    validate="many_to_one",
    indicator=True
)


# ============================================================
# 10. CHECK MERGE RESULTS
# ============================================================

matched = (
    df_final["_merge"] == "both"
).sum()

unmatched = (
    df_final["_merge"] == "left_only"
).sum()

print("\n========== MERGE REPORT ==========")

print("Existing dataset rows:", len(df_original))
print("New dataset rows:", len(df_new))
print("Final dataset rows:", len(df_final))

print("Matched rows:", matched)
print("Unmatched rows:", unmatched)

if len(df_final) != len(df_original):

    raise ValueError(
        "Row count changed unexpectedly. "
        "Check the merge keys."
    )

if unmatched > 0:

    print(
        "\nWARNING: Some existing records did not find "
        "matching cloud-pressure data."
    )

    print("\nSample unmatched records:")

    print(
        df_final.loc[
            df_final["_merge"] == "left_only",
            ["state", "date"]
        ].head(15)
    )


# ============================================================
# 11. CLEAN FINAL DATASET
# ============================================================

df_final.drop(
    columns=[
        "_state_key",
        "_date_key",
        "_merge"
    ],
    inplace=True
)


# Arrange columns in required order

final_columns = [
    "state",
    "lat",
    "lon",
    "date",
    "tmax",
    "tmin",
    "wind_speed",
    "solar_radiation",
    "rh2m",
    "cloud_cover",
    "atmospheric_pressure"
]

df_final = df_final[final_columns]


# ============================================================
# 12. SAVE FINAL DATASET
# ============================================================

df_final.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 13. FINAL DATASET SUMMARY
# ============================================================

print("\n========== FINAL DATASET ==========")

print("Saved successfully!")
print("Output file:", OUTPUT_FILE)

print("Final shape:", df_final.shape)

print("\nColumns:")
print(df_final.columns.tolist())

print("\nMissing values:")
print(
    df_final.isnull().sum().to_string()
)

print("\nFirst five rows:")
print(df_final.head())

print("\nRows per state:")
print(
    df_final.groupby("state")
    .size()
    .to_string()
)

print("\nMerge process completed successfully!")

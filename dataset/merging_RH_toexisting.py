
import os
import pandas as pd

# Current script directory = dataset folder
DATASET_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

# Input files
OLD_FILE = os.path.join(
    DATASET_FOLDER,
    "nasa_power_india_daily.csv"
)

RH_FILE = os.path.join(
    DATASET_FOLDER,
    "nasa_power_india_RH2M_daily.csv"
)

# Output file
OUTPUT_FILE = os.path.join(
    DATASET_FOLDER,
    "nasa_power_india_combined_daily.csv"
)


# --------------------------------------------------
# 2. CHECK FILES
# --------------------------------------------------

for file in [OLD_FILE, RH_FILE]:

    if not os.path.exists(file):
        raise FileNotFoundError(
            f"File not found: {file}"
        )


# --------------------------------------------------
# 3. READ DATASETS
# --------------------------------------------------

print("Reading datasets...")

df_old = pd.read_csv(OLD_FILE)

df_rh = pd.read_csv(RH_FILE)


# --------------------------------------------------
# 4. STANDARDIZE COLUMN NAMES
# --------------------------------------------------

df_old.columns = [
    c.strip().lower() for c in df_old.columns
]

df_rh.columns = [
    c.strip().lower() for c in df_rh.columns
]


# --------------------------------------------------
# 5. CHECK REQUIRED COLUMNS
# --------------------------------------------------

old_required = [
    "state", "lat", "lon", "date",
    "tmax", "tmin", "wind_speed",
    "solar_radiation"
]

rh_required = [
    "state", "date", "rh2m"
]

for col in old_required:

    if col not in df_old.columns:
        raise ValueError(
            f"Missing column in original dataset: {col}"
        )

for col in rh_required:

    if col not in df_rh.columns:
        raise ValueError(
            f"Missing column in RH dataset: {col}"
        )


# --------------------------------------------------
# 6. PREPARE COMMON KEYS
# --------------------------------------------------

df_old["state"] = df_old["state"].astype(str).str.strip()

df_rh["state"] = df_rh["state"].astype(str).str.strip()

df_old["date"] = pd.to_datetime(
    df_old["date"], errors="coerce"
).dt.normalize()

df_rh["date"] = pd.to_datetime(
    df_rh["date"], errors="coerce"
).dt.normalize()


# Remove invalid dates
df_old = df_old.dropna(subset=["date"])

df_rh = df_rh.dropna(subset=["date"])


# --------------------------------------------------
# 7. KEEP REQUIRED RH COLUMNS
# --------------------------------------------------

df_rh = df_rh[
    ["state", "date", "rh2m"]
]


# --------------------------------------------------
# 8. CHECK DUPLICATE KEYS
# --------------------------------------------------

for name, df in [
    ("Original dataset", df_old),
    ("RH2M dataset", df_rh)
]:

    duplicates = df.duplicated(
        subset=["state", "date"]
    ).sum()

    if duplicates > 0:

        raise ValueError(
            f"{name} contains {duplicates} "
            "duplicate state-date records."
        )


# --------------------------------------------------
# 9. MERGE DATASETS
# --------------------------------------------------

print("\nMerging datasets...")

merged_df = pd.merge(
    df_old,
    df_rh,
    on=["state", "date"],
    how="left",
    validate="one_to_one",
    indicator=True
)


# --------------------------------------------------
# 10. CHECK MATCHING
# --------------------------------------------------

matched = (merged_df["_merge"] == "both").sum()

unmatched = (merged_df["_merge"] == "left_only").sum()

print("\nMerge results:")
print("Matched records:", matched)
print("Unmatched records:", unmatched)

if unmatched > 0:

    print("\nWARNING: Some original records have no RH2M match.")

    print(
        merged_df.loc[
            merged_df["_merge"] == "left_only",
            ["state", "date"]
        ].head(10)
    )


# Remove merge indicator
merged_df.drop(
    columns=["_merge"],
    inplace=True
)


# --------------------------------------------------
# 11. FINAL COLUMN ORDER
# --------------------------------------------------

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

merged_df = merged_df[final_columns]


# --------------------------------------------------
# 12. SAVE FINAL DATASET
# --------------------------------------------------

merged_df = merged_df.sort_values(
    ["state", "date"]
).reset_index(drop=True)

merged_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 13. FINAL REPORT
# --------------------------------------------------

print("\n========== MERGE COMPLETED ==========")

print("Output file:", OUTPUT_FILE)

print("Total rows:", len(merged_df))

print("Total columns:", len(merged_df.columns))

print("States/UTs:", merged_df["state"].nunique())

print(
    "Date range:",
    merged_df["date"].min().date(),
    "to",
    merged_df["date"].max().date()
)

print("\nFinal columns:")

print(merged_df.columns.tolist())

print("\nMissing values:")

print(merged_df.isna().sum().to_string())

print("\nRows per state:")

print(
    merged_df.groupby("state").size().to_string()
)

print("\n=====================================")
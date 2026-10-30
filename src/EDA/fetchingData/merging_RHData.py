
import os
import sys
import glob
import pandas as pd


# Input folder
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

# NASA raw data folder
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    PROJECT_ROOT, "nasa_raw_RH2M"
)

print("Project root:", PROJECT_ROOT)
print("NASA data folder:", root)

# Find all CSV files
files = sorted(
    glob.glob(
        os.path.join(root, "**", "*.csv"),
        recursive=True
    )
)

if not files:
    sys.exit(
        f"No CSV files found under '{root}'. "
        "Check the folder path."
    )


# Required columns
NEEDED = ["date", "rh2m"]

frames = []
bad = []


for f in files:

    try:

        df = pd.read_csv(f)

        # Standardize column names
        df.columns = [
            c.strip().lower()
            for c in df.columns
        ]

        # Check required columns
        miss = [
            c for c in NEEDED
            if c not in df.columns
        ]

        if miss:
            bad.append(
                (f, f"missing columns {miss}")
            )
            continue

        # Get state from folder name if unavailable
        if "state" not in df.columns:

            state = os.path.basename(
                os.path.dirname(f)
            ).replace("_", " ")

            df.insert(0, "state", state)

        # Keep only required columns
        df = df[["state", "date", "rh2m"]]

        # Rename RH2M to uppercase
        df = df.rename(
            columns={"rh2m": "RH2M"}
        )

        frames.append(df)

    except Exception as e:

        bad.append(
            (f, f"{type(e).__name__}: {e}")
        )


# Check for usable files
if not frames:

    sys.exit(
        "No usable files. Problems:\n" +
        "\n".join(
            f"  {a}: {b}"
            for a, b in bad
        )
    )


# Combine all files
all_df = pd.concat(
    frames,
    ignore_index=True
)


# Process dates
all_df["date"] = pd.to_datetime(
    all_df["date"],
    errors="coerce"
)

all_df = all_df.dropna(
    subset=["date"]
)


# Process RH2M values
all_df["RH2M"] = pd.to_numeric(
    all_df["RH2M"],
    errors="coerce"
)

# NASA uses -999 for missing values
all_df.loc[
    all_df["RH2M"] == -999,
    "RH2M"
] = float("nan")


# Remove duplicate state-date records
all_df = (
    all_df.drop_duplicates(
        subset=["state", "date"]
    )
    .sort_values(
        ["state", "date"]
    )
    .reset_index(drop=True)
)


# Ensure exactly three columns
all_df = all_df[
    ["state", "date", "RH2M"]
]


# Save final CSV
output_file = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "nasa_power_india_RH2M_daily.csv"
)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

all_df.to_csv(
    output_file,
    index=False
)


# Summary
print("\n========== MERGE COMPLETE ==========")

print(f"Files found: {len(files)}")
print(f"Files successfully read: {len(frames)}")

print(f"\nSaved: {output_file}")

print(f"Total rows: {len(all_df):,}")

print(
    f"States/UTs: {all_df['state'].nunique()}"
)

print(
    f"Dates: {all_df['date'].min().date()} "
    f"to {all_df['date'].max().date()}"
)

print("\nRows per state:")
print(
    all_df.groupby("state")
    .size()
    .to_string()
)

print("\nMissing values per column:")
print(
    all_df.isna().sum().to_string()
)

if bad:

    print("\nFiles skipped:")

    for a, b in bad:
        print(f"  {a}: {b}")

print("\n====================================")

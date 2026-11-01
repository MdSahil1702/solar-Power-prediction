
import os
import json
import time
import random
import shutil
import subprocess
import urllib.request

import requests
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

YEARS = range(2016, 2026)
MAX_PASSES = 40

# Cloud cover and surface atmospheric pressure
PARAMS = "CLOUD_AMT,PS"

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/124 Safari/537.36"
)

CACHE = "nasa_raw_CLOUD_PRESSURE"

OUTPUT_FILE = "nasa_power_india_cloud_pressure_daily.csv"

STATES = {
    "Andhra Pradesh": (15.91, 79.74),
    "Arunachal Pradesh": (28.22, 94.73),
    "Assam": (26.20, 92.94),
    "Bihar": (25.10, 85.31),
    "Chhattisgarh": (21.28, 81.87),
    "Goa": (15.30, 74.12),
    "Gujarat": (22.26, 71.19),
    "Haryana": (29.06, 76.09),
    "Himachal Pradesh": (31.10, 77.17),
    "Jharkhand": (23.61, 85.28),
    "Karnataka": (15.32, 75.71),
    "Kerala": (10.85, 76.27),
    "Madhya Pradesh": (22.97, 78.66),
    "Maharashtra": (19.75, 75.71),
    "Manipur": (24.66, 93.91),
    "Meghalaya": (25.47, 91.37),
    "Mizoram": (23.16, 92.94),
    "Nagaland": (26.16, 94.56),
    "Odisha": (20.95, 85.10),
    "Punjab": (31.15, 75.34),
    "Rajasthan": (27.02, 74.22),
    "Sikkim": (27.53, 88.51),
    "Tamil Nadu": (11.13, 78.66),
    "Telangana": (18.11, 79.02),
    "Tripura": (23.94, 91.99),
    "Uttar Pradesh": (26.85, 80.91),
    "Uttarakhand": (30.07, 79.02),
    "West Bengal": (22.99, 87.85),
    "Delhi": (28.61, 77.21),
    "Jammu and Kashmir": (33.78, 76.58),
    "Ladakh": (34.15, 77.58),
    "Puducherry": (11.94, 79.81),
    "Chandigarh": (30.73, 76.78),
    "Andaman and Nicobar": (11.74, 92.66),
    "Lakshadweep": (10.57, 72.64),
    "Dadra Nagar Haveli and Daman Diu": (20.40, 72.83),
}


# ============================================================
# 1. BUILD NASA POWER URL
# ============================================================

def full_url(lat, lon, year):

    return (
        f"{URL}?parameters={PARAMS}&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={year}0101&end={year}1231"
        f"&format=JSON&time-standard=UTC"
    )


# ============================================================
# 2. DOWNLOAD METHODS
# ============================================================

def via_requests(url):

    r = requests.get(
        url,
        headers={
            "User-Agent": UA,
            "Connection": "close"
        },
        timeout=60
    )

    r.raise_for_status()

    return r.json()


def via_curl(url):

    exe = shutil.which("curl") or shutil.which("curl.exe")

    if not exe:
        raise RuntimeError("curl not found")

    out = subprocess.run(
        [
            exe,
            "-s",
            "-f",
            "--max-time",
            "60",
            "-A",
            UA,
            url
        ],
        capture_output=True,
        text=True,
        timeout=90
    )

    if out.returncode != 0:
        raise RuntimeError(
            f"curl exit {out.returncode}: {out.stderr}"
        )

    return json.loads(out.stdout)


def via_urllib(url):

    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA}
    )

    with urllib.request.urlopen(req, timeout=60) as r:

        return json.loads(r.read().decode())


METHODS = [
    via_requests,
    via_curl,
    via_urllib
]


# ============================================================
# 3. GET NASA JSON WITH FALLBACK METHODS
# ============================================================

def get_json(lat, lon, year):

    url = full_url(lat, lon, year)

    last = None

    for method in random.sample(
        METHODS,
        len(METHODS)
    ):

        try:

            return method(url)

        except Exception as e:

            last = (
                f"{method.__name__}: "
                f"{type(e).__name__}: {e}"
            )

    raise RuntimeError(last)


# ============================================================
# 4. CACHE PATH
# ============================================================

def year_path(state, year):

    folder = os.path.join(
        CACHE,
        state.replace(" ", "_")
    )

    os.makedirs(folder, exist_ok=True)

    return os.path.join(
        folder,
        f"{year}.csv"
    )


# ============================================================
# 5. CONVERT NASA JSON INTO DATAFRAME
# ============================================================

def to_frame(js, state, lat, lon):

    parameter_data = js["properties"]["parameter"]

    cloud_data = parameter_data["CLOUD_AMT"]

    pressure_data = parameter_data["PS"]

    df = pd.DataFrame({

        "date": pd.to_datetime(
            list(cloud_data.keys()),
            format="%Y%m%d"
        ),

        "cloud_cover": list(cloud_data.values()),

        "atmospheric_pressure": [
            pressure_data.get(date, float("nan"))
            for date in cloud_data.keys()
        ]

    })

    df.insert(0, "state", state)

    return df


# ============================================================
# 6. IDENTIFY MISSING DOWNLOADS
# ============================================================

def missing():

    return [
        (state, year)

        for state in STATES

        for year in YEARS

        if not os.path.exists(
            year_path(state, year)
        )
    ]


# ============================================================
# 7. MERGE ALL DOWNLOADED DATA
# ============================================================

def merge():

    frames = []

    for state in STATES:

        for year in YEARS:

            path = year_path(state, year)

            if os.path.exists(path):

                frames.append(
                    pd.read_csv(
                        path,
                        parse_dates=["date"]
                    )
                )

    if not frames:

        print("No downloaded data available.")

        return

    df = pd.concat(
        frames,
        ignore_index=True
    )

    # Convert columns to numeric
    df["cloud_cover"] = pd.to_numeric(
        df["cloud_cover"],
        errors="coerce"
    )

    df["atmospheric_pressure"] = pd.to_numeric(
        df["atmospheric_pressure"],
        errors="coerce"
    )

    # NASA missing-value code
    df.replace(-999, float("nan"), inplace=True)

    # Sort records
    df = df.sort_values(
        ["state", "date"]
    ).reset_index(drop=True)

    # Remove duplicate records
    df = df.drop_duplicates(
        subset=["state", "date"]
    )

    # Keep only required columns
    df = df[
        [
            "state",
            "date",
            "cloud_cover",
            "atmospheric_pressure"
        ]
    ]

    # Save final dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%d-%m-%Y"
    )

    print("\n========================================")
    print("FINAL NASA DATASET SAVED")
    print("========================================")

    print("File:", OUTPUT_FILE)
    print("Shape:", df.shape)

    print(
        "States/UTs:",
        df["state"].nunique()
    )

    print(
        "Date range:",
        df["date"].min(),
        "to",
        df["date"].max()
    )

    print("\nMissing values:\n")

    print(
        df.isna().sum().to_string()
    )

    print("\nDataset preview:")

    print(df.head(10).to_string(index=False))


# ============================================================
# 8. MAIN DOWNLOAD FUNCTION
# ============================================================

def main():

    total = len(STATES) * len(list(YEARS))

    print("NASA POWER CLOUD COVER + PRESSURE DOWNLOADER")
    print("Total states/UTs:", len(STATES))
    print("Years:", list(YEARS))
    print("Total downloads:", total)

    for n in range(1, MAX_PASSES + 1):

        todo = missing()

        if not todo:

            break

        print(
            f"\n=== Pass {n}: {len(todo)} "
            f"of {total} chunks still to download ==="
        )

        for state, year in todo:

            lat, lon = STATES[state]

            try:

                js = get_json(
                    lat,
                    lon,
                    year
                )

                df = to_frame(
                    js,
                    state,
                    lat,
                    lon
                )

                df.to_csv(
                    year_path(state, year),
                    index=False
                )

                print(
                    f"  OK   {state} {year}"
                )

            except Exception as e:

                print(
                    f"  FAIL {state} {year} ({e})"
                )

            time.sleep(
                random.uniform(1.0, 2.5)
            )

        left = len(missing())

        if left == 0:

            break

        wait = min(30 * n, 180)

        print(
            f"{left} chunks left. "
            f"Waiting {wait}s before next pass..."
        )

        time.sleep(wait)

    left = missing()

    merge()

    if left:

        print(
            f"\n{len(left)} chunks still missing. "
            "Run the script again later to resume."
        )

    else:

        print("\nAll downloads completed successfully!")


# ============================================================
# 9. RUN
# ==============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nStopped. Download progress is saved. "
            "Run the script again to continue."
        )

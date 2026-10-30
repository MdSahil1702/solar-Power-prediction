
import os
import json
import time
import random
import shutil
import subprocess
import urllib.request

import requests
import pandas as pd

YEARS = range(2016, 2026)
MAX_PASSES = 40

# Only Relative Humidity at 2 Meters
PARAMS = "RH2M"

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"

CACHE = "nasa_raw_RH2M"

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


def full_url(lat, lon, year):
    return (
        f"{URL}?parameters={PARAMS}&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={year}0101&end={year}1231&format=JSON"
    )


def via_requests(url):
    r = requests.get(
        url,
        headers={"User-Agent": UA, "Connection": "close"},
        timeout=60
    )
    r.raise_for_status()
    return r.json()


def via_curl(url):
    exe = shutil.which("curl") or shutil.which("curl.exe")

    if not exe:
        raise RuntimeError("curl not found")

    out = subprocess.run(
        [exe, "-s", "-f", "--max-time", "60", "-A", UA, url],
        capture_output=True,
        text=True,
        timeout=90
    )

    if out.returncode != 0:
        raise RuntimeError(f"curl exit {out.returncode}")

    return json.loads(out.stdout)


def via_urllib(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA}
    )

    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


METHODS = [via_requests, via_curl, via_urllib]


def get_json(lat, lon, year):
    url = full_url(lat, lon, year)
    last = None

    for method in random.sample(METHODS, len(METHODS)):
        try:
            return method(url)

        except Exception as e:
            last = f"{method.__name__}: {type(e).__name__}"

    raise RuntimeError(last)


def year_path(state, year):
    folder = os.path.join(
        CACHE,
        state.replace(" ", "_")
    )

    os.makedirs(folder, exist_ok=True)

    return os.path.join(folder, f"{year}.csv")


def to_frame(js, state, lat, lon):

    rh_data = js["properties"]["parameter"]["RH2M"]

    df = pd.DataFrame({
        "date": pd.to_datetime(
            list(rh_data.keys()),
            format="%Y%m%d"
        ),
        "RH2M": list(rh_data.values())
    })

    df.insert(0, "state", state)

    return df


def missing():

    return [
        (state, year)
        for state in STATES
        for year in YEARS
        if not os.path.exists(year_path(state, year))
    ]


def merge():

    frames = []

    for state in STATES:
        for year in YEARS:

            path = year_path(state, year)

            if os.path.exists(path):
                frames.append(
                    pd.read_csv(path, parse_dates=["date"])
                )

    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)

    # NASA missing-value code
    df["RH2M"] = pd.to_numeric(
        df["RH2M"], errors="coerce"
    ).replace(-999, float("nan"))

    df = df.sort_values(
        ["state", "date"]
    ).reset_index(drop=True)

    df.to_csv(
        "nasa_power_india_RH2M_daily.csv",
        index=False
    )

    print(
        f"\nSaved nasa_power_india_RH2M_daily.csv:"
        f" {df.shape[0]} rows,"
        f" {df['state'].nunique()} states/UTs"
    )

    print(
        "\nMissing values per column:\n",
        df.isna().sum().to_string()
    )


def main():

    total = len(STATES) * len(list(YEARS))

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

                js = get_json(lat, lon, year)

                df = to_frame(
                    js, state, lat, lon
                )

                df.to_csv(
                    year_path(state, year),
                    index=False
                )

                print(f"  ok   {state} {year}")

            except Exception as e:

                print(
                    f"  fail {state} {year} ({e})"
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
            "Run the script again later."
        )
    else:
        print("\nAll done!")


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nStopped. Progress is saved. "
            "Run again to continue."
        )

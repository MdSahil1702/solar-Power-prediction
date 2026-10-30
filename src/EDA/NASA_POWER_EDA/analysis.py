
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

from pathlib import Path

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parents[3]

# Combined dataset
CSV = PROJECT_DIR / "dataset" / "nasa_power_india_combined_daily.csv"

# Figures output directory
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Project directory:", PROJECT_DIR)
print("Dataset location:", CSV)
print("Dataset exists:", CSV.exists())


# ============================================================
# FEATURES
# ============================================================

VARS = [
    "tmax",
    "tmin",
    "wind_speed",
    "solar_radiation",
    "rh2m"
]

NAME = {
    "tmax": "Max temp (C)",
    "tmin": "Min temp (C)",
    "wind_speed": "Wind (m/s)",
    "solar_radiation": "Radiation (kWh/m2/day)",
    "rh2m": "Relative humidity (%)"
}

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def title(t):
    print(f"\n{'=' * 65}")
    print(t)
    print("=" * 65)


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ============================================================
# LOAD DATA
# ============================================================

title("LOADING DATASET")

if not CSV.exists():
    raise FileNotFoundError(f"Dataset not found: {CSV}")

df = pd.read_csv(CSV)

print("Dataset:", CSV)
print("Shape:", df.shape)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

required_columns = [
    "state", "lat", "lon", "date"
] + VARS

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Parse dates
df["date"] = pd.to_datetime(
    df["date"].astype(str).str.strip(),
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

if df["date"].isna().any():
    raise ValueError(
        f"Invalid dates found: {df['date'].isna().sum()}"
    )

df = df.sort_values(
    ["state", "date"]
).reset_index(drop=True)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# 10-day period of year
df["dk"] = (
    (df["month"] - 1) * 3
    + np.where(
        df["date"].dt.day <= 10,
        0,
        np.where(df["date"].dt.day <= 20, 1, 2)
    )
)

# Temperature range
df["range"] = df["tmax"] - df["tmin"]

# State-wise summary
state = (
    df.groupby("state")
    .agg(
        lat=("lat", "first"),
        **{v: (v, "mean") for v in VARS}
    )
    .sort_values("lat")
)


# ============================================================
# 1. DATA HEALTH
# ============================================================

title("1. DATA HEALTH")

days = df.groupby("state").size()

print(
    f"Rows: {len(df)} | "
    f"States/UTs: {df['state'].nunique()} | "
    f"Date range: {df['date'].min().date()} "
    f"to {df['date'].max().date()}"
)

print(f"Days per state: {days.min()} to {days.max()}")

print("\nMissing values:")
print(df[VARS].isna().sum())

print("\nInvalid physical values:")
print(
    "Tmin > Tmax:",
    int((df["tmin"] > df["tmax"]).sum())
)

print(
    "Negative radiation:",
    int((df["solar_radiation"] < 0).sum())
)

print(
    "Negative wind speed:",
    int((df["wind_speed"] < 0).sum())
)

print(
    "RH2M outside 0-100:",
    int(((df["rh2m"] < 0) | (df["rh2m"] > 100)).sum())
)

print("\nDescriptive statistics:")
print(
    df[VARS]
    .describe()
    .T[["min", "mean", "max", "std"]]
    .round(2)
)


# ============================================================
# 2. SEASONAL PATTERN
# ============================================================

title("2. SEASONAL PATTERN")

# Seasonal heatmaps
fig, ax = plt.subplots(3, 2, figsize=(16, 16))

for a, v in zip(ax.ravel(), VARS):

    m = (
        df.groupby(["state", "dk"])[v]
        .mean()
        .unstack()
        .reindex(state.index)
    )

    im = a.imshow(
        m.values,
        aspect="auto",
        cmap="viridis"
    )

    plt.colorbar(im, ax=a)

    a.set_yticks(range(len(m)))
    a.set_yticklabels(m.index, fontsize=6)

    a.set_xticks(range(0, 36, 3))
    a.set_xticklabels(MONTHS)

    a.set_title(NAME[v] + " (states south to north)")

# Hide unused subplot
ax.ravel()[-1].axis("off")

save("1_seasonal_heatmaps")


# National seasonal pattern
nat = df.groupby("dk")[VARS].mean()
mon = df.groupby("month")[VARS].mean()

print("\nNational seasonal peaks and lows:")

for v in VARS:

    print(
        f"{NAME[v]:28s} "
        f"peak period {nat[v].idxmax():2d} "
        f"({nat[v].max():.2f}) | "
        f"low period {nat[v].idxmin():2d} "
        f"({nat[v].min():.2f})"
    )

print(
    "\nBest radiation month:",
    MONTHS[mon["solar_radiation"].idxmax() - 1]
)

print(
    "Worst radiation month:",
    MONTHS[mon["solar_radiation"].idxmin() - 1]
)


# Monsoon versus summer radiation
mm = (
    df.groupby(["state", "month"])["solar_radiation"]
    .mean()
    .unstack()
)

dip = (
    mm[[6, 7, 8]].mean(axis=1)
    / mm[[3, 4, 5]].mean(axis=1)
).sort_values()

print("\nMonsoon / summer radiation ratio:")
print("Lowest ratios:")
print(dip.round(2).head(5).to_string())

print("\nHighest ratios:")
print(dip.round(2).tail(3).to_string())


# Seasonal line plots
selected_states = [
    "Rajasthan",
    "Gujarat",
    "Kerala",
    "Assam",
    "Himachal Pradesh",
    "Tamil Nadu"
]

fig, ax = plt.subplots(2, 2, figsize=(15, 9))

LINE_VARS = [
    "solar_radiation",
    "tmax",
    "wind_speed",
    "rh2m"
]

for a, v in zip(ax.ravel(), LINE_VARS):

    for s in selected_states:

        state_data = df[df["state"] == s]

        if not state_data.empty:

            seasonal = (
                state_data.groupby("dk")[v]
                .mean()
            )

            a.plot(
                seasonal.index,
                seasonal.values,
                label=s
            )

    a.set_title(NAME[v])
    a.set_xlabel("Period of year (0 = 1-10 Jan)")
    a.set_ylabel(NAME[v])
    a.grid(alpha=0.3)

ax[0, 0].legend(fontsize=7)

save("2_seasonal_lines")


# ============================================================
# 3. RELATIONS BETWEEN FEATURES
# ============================================================

title("3. FEATURE RELATIONS")

cols = VARS + ["range"]

print("\nRaw correlation:")
print(df[cols].corr().round(2))


# Remove state-wise seasonal effects
anom = (
    df[cols]
    - df.groupby(["state", "dk"])[cols]
    .transform("mean")
)

print("\nCorrelation after removing seasonality:")
print(anom.corr().round(2))


# Correlation heatmaps
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

for a, (data, t) in zip(
    ax,
    [
        (df[cols], "Raw correlation"),
        (anom, "Correlation without seasonality")
    ]
):

    c = data.corr()

    im = a.imshow(
        c.values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(im, ax=a)

    a.set_title(t)

    a.set_xticks(range(len(cols)))
    a.set_xticklabels(cols, rotation=45, ha="right")

    a.set_yticks(range(len(cols)))
    a.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):

            a.text(
                j,
                i,
                f"{c.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )


save("3_correlation_heatmaps")


# Scatter plots
s = df.sample(
    min(20000, len(df)),
    random_state=1
)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))

ax[0].scatter(
    s["solar_radiation"],
    s["range"],
    s=3,
    alpha=0.3,
    c=s["month"],
    cmap="twilight"
)

ax[0].set_xlabel("Solar radiation")
ax[0].set_ylabel("Tmax - Tmin")
ax[0].set_title("Radiation vs temperature range")


ax[1].scatter(
    s["rh2m"],
    s["solar_radiation"],
    s=3,
    alpha=0.3,
    c=s["month"],
    cmap="viridis"
)

ax[1].set_xlabel("Relative humidity (%)")
ax[1].set_ylabel("Solar radiation")
ax[1].set_title("Humidity vs solar radiation")


ax[2].scatter(
    s["rh2m"],
    s["tmax"],
    s=3,
    alpha=0.3,
    c=s["month"],
    cmap="plasma"
)

ax[2].set_xlabel("Relative humidity (%)")
ax[2].set_ylabel("Maximum temperature")
ax[2].set_title("Humidity vs maximum temperature")

save("4_feature_scatter_plots")


# Seasonal correlations
for name, ms in {
    "winter": [12, 1, 2],
    "summer": [3, 4, 5],
    "monsoon": [6, 7, 8, 9],
    "post-monsoon": [10, 11]
}.items():

    d = df[df.month.isin(ms)]

    print(
        f"\n{name.upper()} correlations:"
    )

    print(
        "Radiation vs temperature range:",
        round(
            d["solar_radiation"].corr(d["range"]),
            3
        )
    )

    print(
        "Radiation vs RH2M:",
        round(
            d["solar_radiation"].corr(d["rh2m"]),
            3
        )
    )

    print(
        "Temperature vs RH2M:",
        round(
            d["tmax"].corr(d["rh2m"]),
            3
        )
    )


# ============================================================
# 4. STATE-WISE ANALYSIS
# ============================================================

title("4. STATE-WISE ANALYSIS")

print(
    "\nSolar radiation top:",
    state["solar_radiation"].nlargest(3).round(2).to_dict()
)

print(
    "Solar radiation low:",
    state["solar_radiation"].nsmallest(3).round(2).to_dict()
)

print(
    "\nTmax hottest:",
    state["tmax"].nlargest(3).round(2).to_dict()
)

print(
    "Tmax coldest:",
    state["tmax"].nsmallest(3).round(2).to_dict()
)

print(
    "\nRH2M highest:",
    state["rh2m"].nlargest(3).round(2).to_dict()
)

print(
    "RH2M lowest:",
    state["rh2m"].nsmallest(3).round(2).to_dict()
)


print(
    "\nLatitude vs Tmax correlation:",
    round(state.lat.corr(state.tmax), 3)
)

print(
    "Latitude vs radiation correlation:",
    round(state.lat.corr(state.solar_radiation), 3)
)

print(
    "Latitude vs RH2M correlation:",
    round(state.lat.corr(state.rh2m), 3)
)


# State-wise plots
order = state.solar_radiation.sort_values().index

fig, ax = plt.subplots(2, 2, figsize=(16, 10))

box = [
    df[df.state == s_].solar_radiation.values
    for s_ in order
]

try:
    ax[0, 0].boxplot(
        box,
        vert=False,
        tick_labels=order,
        showfliers=False
    )

except TypeError:
    ax[0, 0].boxplot(
        box,
        vert=False,
        labels=order,
        showfliers=False
    )

ax[0, 0].tick_params(axis="y", labelsize=6)
ax[0, 0].set_title("Solar radiation by state")


ax[0, 1].scatter(
    state.lat,
    state.solar_radiation
)

ax[0, 1].set_xlabel("Latitude")
ax[0, 1].set_ylabel("Solar radiation")
ax[0, 1].set_title("Latitude vs mean radiation")


ax[1, 0].scatter(
    state.lat,
    state.tmax
)

ax[1, 0].set_xlabel("Latitude")
ax[1, 0].set_ylabel("Maximum temperature")
ax[1, 0].set_title("Latitude vs mean Tmax")


ax[1, 1].scatter(
    state.lat,
    state.rh2m
)

ax[1, 1].set_xlabel("Latitude")
ax[1, 1].set_ylabel("Relative humidity (%)")
ax[1, 1].set_title("Latitude vs mean RH2M")

save("5_state_analysis")


# ============================================================
# 5. YEARLY TREND
# ============================================================

title("5. YEARLY TREND (10 YEARS ONLY - NOT CLIMATE CHANGE)")

yr = df.groupby("year")[VARS].mean()

print(yr.round(2))

fig, ax = plt.subplots(2, 3, figsize=(16, 8))

for a, v in zip(ax.ravel(), VARS):

    a.plot(
        yr.index,
        yr[v],
        marker="o"
    )

    a.set_title(NAME[v])
    a.set_xlabel("Year")
    a.grid(alpha=0.3)

ax.ravel()[-1].axis("off")

save("6_yearly_trends")


# ============================================================
# 6. DISTRIBUTIONS
# ============================================================

title("6. FEATURE DISTRIBUTIONS")

print(
    "\nFeature skewness:"
)

print(
    df[VARS].skew().round(2).to_dict()
)

fig, ax = plt.subplots(2, 3, figsize=(16, 8))

for a, v in zip(ax.ravel(), VARS):

    a.hist(
        df[v].dropna(),
        bins=50
    )

    a.set_title(NAME[v])
    a.set_xlabel(NAME[v])
    a.set_ylabel("Frequency")

ax.ravel()[-1].axis("off")

save("7_feature_distributions")


# ============================================================
# 7. CLIMATOLOGY BASELINE
# ============================================================

title(
    "7. CLIMATOLOGY BASELINE "
    "(TRAIN <= 2022, TEST > 2022, 10-DAY MEANS)"
)

# Average values for each state, year and 10-day period
tab = (
    df.groupby(["state", "year", "dk"], as_index=False)[VARS]
    .mean()
)

# Training and testing
tr = tab[tab.year <= 2022].copy()
te = tab[tab.year > 2022].copy()

# Climatology from training period
climatology = (
    tr.groupby(["state", "dk"])[VARS]
    .mean()
    .add_suffix("_c")
)

base = te.join(
    climatology,
    on=["state", "dk"]
)

# Calculate metrics
results = {}

for v in VARS:

    actual = base[v]
    predicted = base[v + "_c"]

    valid = actual.notna() & predicted.notna()

    actual = actual[valid]
    predicted = predicted[valid]

    mae = (actual - predicted).abs().mean()

    ss_res = ((actual - predicted) ** 2).sum()

    ss_tot = ((actual - actual.mean()) ** 2).sum()

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot != 0
        else np.nan
    )

    results[v] = {
        "MAE": mae,
        "R2": r2
    }

res = pd.DataFrame(results).T

print("\nClimatology baseline results:")
print(res.round(3))

print("\nNumber of test observations:", len(te))
print(
    "Observations with climatology available:",
    base[VARS[0] + "_c"].notna().sum()
)


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

title("8. EDA COMPLETED")

print("Total records:", len(df))
print("Total states/UTs:", df["state"].nunique())

print(
    "Date range:",
    df["date"].min().date(),
    "to",
    df["date"].max().date()
)

print("\nFeatures analyzed:")

for v in VARS:
    print("-", NAME[v])

print("\nFigures saved in:")
print(FIGURES_DIR)

print("\nEDA and visualization completed successfully!")

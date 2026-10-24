"""NASA POWER data analysis. Run: python nasa_analysis.py [csv_path]
Prints the key tables and saves charts to ./figures. Needs: pandas numpy matplotlib"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

CSV = PROJECT_DIR / "dataset" / "nasa_power_india_daily.csv"
OUTPUT_DIR = PROJECT_DIR / "analysis_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARS = ["tmax", "tmin", "wind_speed", "solar_radiation"]
NAME = {"tmax": "Max temp (C)", "tmin": "Min temp (C)", "wind_speed": "Wind (m/s)", "solar_radiation": "Radiation (kWh/m2/day)"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
os.makedirs("figures", exist_ok=True)


def title(t):
    print(f"\n=== {t} ===")


def save(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=150)
    plt.show()
    plt.close()


# ---------- load ----------
df = pd.read_csv(CSV)
iso = str(df["date"].iloc[0])[4:5] == "-"                      # 2016-01-31 or 31-01-2016
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d" if iso else "%d-%m-%Y")
df = df.sort_values(["state", "date"]).reset_index(drop=True)
df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
df["dk"] = (df["month"] - 1) * 3 + np.where(df["date"].dt.day <= 10, 0, np.where(df["date"].dt.day <= 20, 1, 2))
df["range"] = df["tmax"] - df["tmin"]
state = df.groupby("state").agg(lat=("lat", "first"), **{v: (v, "mean") for v in VARS}).sort_values("lat")

# ---------- 1. data health ----------
title("1. Data health")
days = df.groupby("state").size()
print(f"rows {len(df)} | states {df['state'].nunique()} | {df['date'].min().date()} to {df['date'].max().date()}")
print(f"days per state: {days.min()} to {days.max()}")
print("missing:", df[VARS].isna().sum().to_dict())
print("tmin>tmax:", int((df.tmin > df.tmax).sum()), "| radiation<0:", int((df.solar_radiation < 0).sum()))
print(df[VARS].describe().T[["min", "mean", "max", "std"]].round(2))

# ---------- 2. seasonal pattern ----------
title("2. Seasonal pattern")
fig, ax = plt.subplots(2, 2, figsize=(15, 11))
for a, v in zip(ax.ravel(), VARS):
    m = df.groupby(["state", "dk"])[v].mean().unstack().loc[state.index]
    plt.colorbar(a.imshow(m.values, aspect="auto", cmap="viridis"), ax=a)
    a.set_yticks(range(len(m))); a.set_yticklabels(m.index, fontsize=6)
    a.set_xticks(range(0, 36, 3)); a.set_xticklabels(MONTHS); a.set_title(NAME[v] + " (states south to north)")
save("1_seasonal_heatmaps")
nat = df.groupby("dk")[VARS].mean()
mon = df.groupby("month")[VARS].mean()
for v in VARS:
    print(f"{NAME[v]:24s} peak period {nat[v].idxmax():2d} ({nat[v].max():.2f}) | low period {nat[v].idxmin():2d} ({nat[v].min():.2f})")
print(f"radiation: best month {MONTHS[mon.solar_radiation.idxmax() - 1]}, worst {MONTHS[mon.solar_radiation.idxmin() - 1]}")

mm = df.groupby(["state", "month"])["solar_radiation"].mean().unstack()
dip = (mm[[6, 7, 8]].mean(axis=1) / mm[[3, 4, 5]].mean(axis=1)).sort_values()
print("monsoon / summer radiation ratio (lowest = biggest dip):")
print(dip.round(2).head(5).to_string()); print("...highest:"); print(dip.round(2).tail(3).to_string())
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for a, v in zip(ax, ["solar_radiation", "tmax", "wind_speed"]):
    for s in ["Rajasthan", "Gujarat", "Kerala", "Assam", "Himachal Pradesh", "Tamil Nadu"]:
        a.plot(df[df.state == s].groupby("dk")[v].mean().values, label=s)
    a.set_title(NAME[v]); a.set_xlabel("period of year (0 = 1-10 Jan)"); a.grid(alpha=.3)
ax[0].legend(fontsize=7)
save("2_seasonal_lines")

# ---------- 3. relations between features ----------
title("3. Feature relations")
cols = VARS + ["range"]
print("raw correlation:\n", df[cols].corr().round(2))
anom = df[cols] - df.groupby(["state", "dk"])[cols].transform("mean")
print("\nafter removing seasonality (state x period mean):\n", anom.corr().round(2))
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, (d, t) in zip(ax[:2], [(df[cols], "Raw correlation"), (anom, "Correlation without seasonality")]):
    c = d.corr(); a.imshow(c.values, cmap="coolwarm", vmin=-1, vmax=1); a.set_title(t)
    a.set_xticks(range(5)); a.set_xticklabels(cols, rotation=45, ha="right"); a.set_yticks(range(5)); a.set_yticklabels(cols)
    for i in range(5):
        for j in range(5): a.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
s = df.sample(20000, random_state=1)
ax[2].scatter(s.solar_radiation, s.range, s=2, alpha=.3, c=s.month, cmap="twilight")
ax[2].set_xlabel("radiation"); ax[2].set_ylabel("Tmax - Tmin"); ax[2].set_title("Radiation vs day-night range")
save("3_relations")
for name, ms in {"winter": [12, 1, 2], "summer": [3, 4, 5], "monsoon": [6, 7, 8, 9], "post-monsoon": [10, 11]}.items():
    d = df[df.month.isin(ms)]
    print(f"corr(radiation, range) {name:13s}: {d.solar_radiation.corr(d.range):.2f}")

# ---------- 4. states ----------
title("4. States")
print("radiation  top:", state.solar_radiation.nlargest(3).round(2).to_dict())
print("radiation  low:", state.solar_radiation.nsmallest(3).round(2).to_dict())
print("Tmax       hot:", state.tmax.nlargest(3).round(1).to_dict(), "| cold:", state.tmax.nsmallest(3).round(1).to_dict())
print(f"corr(latitude, Tmax) {state.lat.corr(state.tmax):.2f} | corr(latitude, radiation) {state.lat.corr(state.solar_radiation):.2f}")
order = state.solar_radiation.sort_values().index
fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
box = [df[df.state == s_].solar_radiation.values for s_ in order]
try: ax[0].boxplot(box, vert=False, tick_labels=order, showfliers=False)
except TypeError: ax[0].boxplot(box, vert=False, labels=order, showfliers=False)
ax[0].tick_params(axis="y", labelsize=6); ax[0].set_title("Radiation by state")
ax[1].scatter(state.lat, state.solar_radiation); ax[1].set_xlabel("latitude"); ax[1].set_title("Latitude vs mean radiation")
ax[2].scatter(state.lat, state.tmax); ax[2].set_xlabel("latitude"); ax[2].set_title("Latitude vs mean Tmax")
save("4_states")

# ---------- 5. trend ----------
title("5. Trend by year (10 years only: do not read as climate change)")
yr = df.groupby("year")[VARS].mean()
print(yr.round(2))
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for a, v in zip(ax, VARS):
    a.plot(yr.index, yr[v], marker="o"); a.set_title(NAME[v]); a.grid(alpha=.3)
save("5_yearly_trend")

# ---------- 6. distributions ----------
title("6. Distributions")
print("skew:", df[VARS].skew().round(2).to_dict())
fig, ax = plt.subplots(1, 4, figsize=(17, 3.5))
for a, v in zip(ax, VARS):
    a.hist(df[v], bins=50); a.set_title(NAME[v])
save("6_distributions")

# ---------- 7. baseline for Stage 1 (what ML has to beat) ----------
title("7. Climatology baseline (train <= 2022, test > 2022, 10-day means)")
tab = df.groupby(["state", "year", "dk"], as_index=False)[VARS].mean()
tr, te = tab[tab.year <= 2022], tab[tab.year > 2022]
base = te.join(tr.groupby(["state", "dk"])[VARS].mean().add_suffix("_c"), on=["state", "dk"])
res = pd.DataFrame({v: {"MAE": (base[v] - base[v + "_c"]).abs().mean(),
                        "R2": 1 - ((base[v] - base[v + "_c"]) ** 2).sum() / ((base[v] - base[v].mean()) ** 2).sum()} for v in VARS}).T
print(res.round(3))
print("\nFigures saved in ./figures")
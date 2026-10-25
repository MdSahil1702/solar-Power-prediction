
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]

CSV = PROJECT_DIR / "dataset" / "Solar power data"/"spg.csv"

RENAME = {"temperature_2_m_above_gnd": "temp", "relative_humidity_2_m_above_gnd": "humidity",
          "wind_speed_10_m_above_gnd": "wind", "angle_of_incidence": "aoi",
          "shortwave_radiation_backwards_sfc": "radiation", "total_cloud_cover_sfc": "cloud",
          "zenith": "zenith", "azimuth": "azimuth", "generated_power_kw": "power"}
os.makedirs("figures", exist_ok=True)
title = lambda t: print(f"\n=== {t} ===")


def save(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=150)
    plt.show()
    plt.close()


df = pd.read_csv(CSV).rename(columns=RENAME)
df = df[[c for c in RENAME.values() if c in df.columns]]
n = len(df)
df["cos_aoi"] = np.clip(np.cos(np.radians(df["aoi"])), 0, None)
FEATS = [c for c in ["temp", "humidity", "wind", "radiation", "cloud", "aoi"] if c in df.columns]

# ---------- 1. data health ----------
title("1. Data health")
print(f"rows {n} | columns {list(df.columns[:-1])}")
print("missing:", int(df.isna().sum().sum()), "| duplicate rows:", int(df.duplicated().sum()))
print(df.drop(columns="cos_aoi").describe().T[["min", "mean", "max", "std"]].round(2))
for need in ["radiation", "cloud"]:
    if need not in df.columns:
        print(f"WARNING: no '{need}' column. Radiation and cloud are the main drivers of power.")
if df["wind"].max() > 30:
    print(f"NOTE: wind max {df['wind'].max():.0f}: values look like km/h, not m/s. Divide by 3.6 to match NASA WS10M.")
behind = df["aoi"] > 90
print(f"rows with AOI > 90 (sun behind panel): {behind.sum()} ({behind.mean():.1%}) | mean power there {df.loc[behind, 'power'].mean():.0f} kW"
      f" (not zero: diffuse light) vs {df.loc[~behind, 'power'].mean():.0f} kW otherwise")

fig, ax = plt.subplots(1, len(df.columns) - 1, figsize=(3.4 * (len(df.columns) - 1), 3.3))
for a, c in zip(ax, [c for c in df.columns if c != "cos_aoi"]):
    a.hist(df[c], bins=40); a.set_title(c)
save("1_distributions")

# ---------- 2. relations with power ----------
title("2. Relations with power")
num = df[FEATS + ["cos_aoi", "power"]]
rel = pd.DataFrame({"pearson": num.corr()["power"], "spearman": num.corr(method="spearman")["power"]}).drop("power")
print(rel.round(2))
print("feature-to-feature (watch strong links):"); print(df[FEATS].corr().round(2))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
c = num.corr(); ax[0].imshow(c.values, cmap="coolwarm", vmin=-1, vmax=1)
ax[0].set_xticks(range(len(c))); ax[0].set_xticklabels(c.columns, rotation=45, ha="right"); ax[0].set_yticks(range(len(c))); ax[0].set_yticklabels(c.columns)
for i in range(len(c)):
    for j in range(len(c)): ax[0].text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax[0].set_title("Correlation")
ax[1].scatter(df["aoi"], df["power"], s=4, alpha=.4); ax[1].set_xlabel("angle of incidence (deg)"); ax[1].set_ylabel("power (kW)"); ax[1].set_title("Power vs angle of incidence")
save("2_relations")

# ---------- 3. power vs angle of incidence ----------
title("3. Power by angle-of-incidence bin")
g = df.groupby(pd.cut(df["aoi"], [0, 15, 25, 35, 45, 55, 65, 75, 90, 180]), observed=True)["power"].agg(["mean", "std", "count"]).round(0)
print(g)
print("Large std inside a bin = something other than the sun angle (clouds) drives power.")
if "radiation" in df.columns:
    df["g_cos"] = df["radiation"] * df["cos_aoi"]
    print(f"corr(power, radiation) {df.power.corr(df.radiation):.3f} | corr(power, radiation x cos AOI) {df.power.corr(df.g_cos):.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(df.radiation, df.power, c=df.aoi, s=5); ax[0].set_xlabel("radiation"); ax[0].set_title("Power vs radiation (colour = AOI)")
    ax[1].scatter(df.g_cos, df.power, s=5, alpha=.5); ax[1].set_xlabel("radiation x cos(AOI)"); ax[1].set_title("Engineered feature")
    save("3_radiation")

# ---------- 4. effect of other factors at similar sun angle ----------
title("4. Other factors, with sun angle held roughly fixed (AOI < 50)")
s = df[df["aoi"] < 50].copy(); s["ratio"] = s["power"] / np.cos(np.radians(s["aoi"]))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, (col, bins) in zip(ax, [("humidity", [0, 20, 35, 50, 70, 101]), ("temp", [-6, 0, 5, 10, 15, 20, 25, 35]), ("wind", [-1, 5, 10, 15, 20, 30, 70])]):
    m = s.groupby(pd.cut(s[col], bins), observed=True)["ratio"].mean()
    print(f"power/cos(AOI) by {col}:\n{m.round(0).to_string()}")
    a.bar(range(len(m)), m.values); a.set_xticks(range(len(m))); a.set_xticklabels([str(i) for i in m.index], rotation=45, fontsize=7); a.set_title(f"power/cos(AOI) by {col}")
save("4_other_factors")
print("Careful: temperature rises with season and sunshine, so its raw effect is mixed up with sun height and clouds.")

# ---------- 5. order in the file ----------
title("5. Order in the file and season coverage")
for k in range(5):
    c = df.iloc[k * n // 5:(k + 1) * n // 5]
    print(f"part {k + 1}: temp {c.temp.mean():5.1f} | humidity {c.humidity.mean():3.0f} | power {c.power.mean():5.0f}" if "humidity" in c else f"part {k + 1}: power {c.power.mean():.0f}")
fig, ax = plt.subplots(2, 1, figsize=(13, 6))
ax[0].plot(df["power"].values[:200]); ax[0].set_title("First 200 rows: daily humps show hourly daytime order")
ax[1].plot(df["power"].rolling(100).mean().values); ax[1].set_title("Power, 100-row rolling mean (season and weather)")
save("5_order")

# ---------- 6. quick model check (chronological vs random split) ----------
title("6. Quick model check")
F = FEATS + ["cos_aoi"]; cut = int(n * .8)
rf = lambda: RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0)
res = []
for name, m in [("Linear Regression", LinearRegression()), ("Random Forest", rf())]:
    m.fit(df[F][:cut], df.power[:cut]); p = np.clip(m.predict(df[F][cut:]), 0, None)
    res.append((name + " (time split)", r2_score(df.power[cut:], p), mean_absolute_error(df.power[cut:], p)))
Xa, Xb, ya, yb = train_test_split(df[F], df.power, test_size=.2, random_state=0)
p = rf().fit(Xa, ya).predict(Xb); res.append(("Random Forest (random split, inflated)", r2_score(yb, p), mean_absolute_error(yb, p)))
print(pd.DataFrame(res, columns=["model", "R2", "MAE kW"]).round(3).to_string(index=False))
m = rf().fit(df[F][:cut], df.power[:cut])
imp = permutation_importance(m, df[F][cut:], df.power[cut:], n_repeats=10, random_state=0).importances_mean
print("permutation importance (time split):", {k: round(float(v), 3) for k, v in zip(F, imp)})

# ---------- 7. panel geometry check (needs zenith and azimuth) ----------
if {"zenith", "azimuth"} <= set(df.columns):
    title("7. Panel tilt and orientation fitted from the data")
    from scipy.optimize import least_squares
    d = df[df.zenith < 90]; z, a, ao = [np.radians(d[c].values) for c in ("zenith", "azimuth", "aoi")]
    f = lambda p: np.arccos(np.clip(np.cos(z) * np.cos(np.radians(p[0])) + np.sin(z) * np.sin(np.radians(p[0])) * np.cos(a - np.radians(p[1])), -1, 1))
    r = least_squares(lambda p: f(p) - ao, [40, 180])
    print(f"tilt {r.x[0]:.1f} deg, azimuth {r.x[1]:.1f} deg, RMS error {np.degrees(np.sqrt(np.mean((f(r.x) - ao) ** 2))):.2f} deg")
print("\nFigures saved in ./figures")
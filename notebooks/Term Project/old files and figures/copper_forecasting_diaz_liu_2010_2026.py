"""Copper price forecasting inspired by Díaz et al. (2020) and Liu et al. (2017).

Place this script in the same folder as the eight Investing.com CSV files whose
names end in '(1).csv'. Run: python copper_forecasting_diaz_liu_2010_2026.py

This is an educational implementation, not an exact replication of either paper.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=RuntimeWarning)
BASE = Path(__file__).resolve().parent
OUT = BASE / "copper_forecasting_outputs"
OUT.mkdir(exist_ok=True)
START_DATE, END_DATE = "2010-01-01", "2026-10-13"
FILES = {
    "Copper": "Copper Futures Historical Data(1).csv",
    "Gold": "Gold Futures Historical Data(1).csv",
    "Silver": "Silver Futures Historical Data(1).csv",
    "Oil": "Crude Oil WTI Futures Historical Data(1).csv",
    "Gas": "Natural Gas Futures Historical Data(1).csv",
    "Lean Hogs": "Lean Hogs Futures Historical Data(1).csv",
    "Coffee": "US Coffee C Futures Historical Data(1).csv",
    "Dow Jones": "Dow Jones Industrial Average Historical Data(1).csv",
}

# 1. Load data and align all series on common dates
frames = []
for name, filename in FILES.items():
    path = BASE / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {filename}. Put all eight CSVs next to this script.")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if not {"Date", "Price"}.issubset(df.columns):
        raise ValueError(f"{filename} must contain 'Date' and 'Price' columns; got {df.columns.tolist()}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")
    df = df[["Date", "Price"]].dropna().drop_duplicates("Date").rename(columns={"Price": name})
    frames.append(df)
data = frames[0]
for frame in frames[1:]:
    data = data.merge(frame, on="Date", how="inner")
data = data.sort_values("Date").set_index("Date").loc[START_DATE:END_DATE].dropna()
if len(data) < 100:
    raise ValueError(f"Only {len(data)} common observations found. Check the CSV files and date overlap.")
data.to_csv(OUT / "aligned_price_data_2010_2026.csv")
vars_ = list(FILES)
print(f"Aligned observations: {len(data):,} | {data.index.min().date()} to {data.index.max().date()}")

# 2. Table 1: descriptive statistics and correlations
stats = pd.DataFrame({"Mean": data.mean(), "Std. Dev.": data.std(ddof=1), "Min": data.min(), "Max": data.max()}).round(2)
stats.to_csv(OUT / "table1_panel_A_descriptive_statistics.csv")
corr = data.corr()
pvals = pd.DataFrame(np.ones(corr.shape), index=vars_, columns=vars_)
for i, a in enumerate(vars_):
    for j, b in enumerate(vars_):
        if i != j:
            _, pvals.loc[a, b] = pearsonr(data[a], data[b])
corr_text = pd.DataFrame("", index=vars_, columns=vars_)
for i, a in enumerate(vars_):
    for j, b in enumerate(vars_):
        if j <= i:
            corr_text.loc[a, b] = "1.00" if i == j else f"{corr.loc[a,b]:.2f}{'*' if pvals.loc[a,b] < .05 else ''}"
corr_text.to_csv(OUT / "table1_panel_B_correlations.csv")
fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 1.5]})
axes[0].axis("off")
tab = axes[0].table(cellText=np.column_stack([stats.index, stats.values]), colLabels=["Variable", *stats.columns], loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.5)
axes[0].set_title("Panel A: Descriptive Statistics", loc="left", fontweight="bold")
axes[1].axis("off")
tab2 = axes[1].table(cellText=np.column_stack([corr_text.index, corr_text.values]), colLabels=["Variable", *vars_], loc="center", cellLoc="center")
tab2.auto_set_font_size(False); tab2.set_fontsize(8); tab2.scale(1, 1.5)
axes[1].set_title("Panel B: Correlations (* p < 0.05)", loc="left", fontweight="bold")
fig.suptitle("Table 1. Descriptive Statistics and Correlations (2010–2026)", y=.99)
fig.tight_layout(); fig.savefig(OUT / "table1_descriptive_statistics_and_correlations.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# 3. Figure 1: all eight price series
fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
for ax, col in zip(axes.ravel(), vars_):
    ax.plot(data.index, data[col], linewidth=.8)
    ax.set_title(col); ax.set_ylabel("Price / index")
    ax.grid(alpha=.25)
fig.suptitle("Figure 1. Price Series (2010–2026)")
fig.tight_layout(); fig.savefig(OUT / "figure1_price_series.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 4. Figure 2: copper vs. other price series
others = [v for v in vars_ if v != "Copper"]
fig, axes = plt.subplots(3, 2, figsize=(12, 13))
for ax, col in zip(axes.ravel(), others):
    ax.scatter(data[col], data["Copper"], s=7, alpha=.35)
    ax.set_xlabel(col); ax.set_ylabel("Copper price"); ax.set_title(f"Copper vs. {col}"); ax.grid(alpha=.2)
fig.suptitle("Figure 2. Copper Price vs. Explanatory Variables")
fig.tight_layout(); fig.savefig(OUT / "figure2_copper_scatterplots.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 5. Figure 3: copper correlations with explanatory variables
cross = corr.loc[others, "Copper"].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5)); cross.plot(kind="bar", ax=ax)
ax.set_ylabel("Pearson correlation coefficient"); ax.set_title("Figure 3. Correlation with Copper Price"); ax.axhline(0, linewidth=.8); ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(OUT / "figure3_copper_correlations.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 6. Forecast evaluation. Horizon labels are approximate trading-day horizons.
# Daily prices are forward-filled only through the target construction; merged data
# remain common-date observations. Horizon: 1d=1, 1w=5, 1m=21, 6m=126, 1y=252, 2y=504.
HORIZONS = {"1 day": 1, "1 week": 5, "1 month": 21, "6 months": 126, "1 year": 252, "2 years": 504}
LAGS = [1, 5, 10, 20, 40, 80]
MODEL_NAMES = ["RT", "RF", "GBM", "RW"]

def make_model(name, seed=42):
    if name == "RT":
        return DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=seed)
    if name == "RF":
        return RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_leaf=4, max_features=0.8, n_jobs=-1, random_state=seed)
    if name == "GBM":
        return GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.04, loss="squared_error", random_state=seed)
    raise ValueError(name)

# Out-of-sample split: first 80% train, final 20% test; no shuffling.
# For each forecast horizon and D, features are D lags of copper and each external series.
# To keep the runtime reasonable, each model is fitted once per (D, horizon).
results = []
price = data["Copper"]
N = len(data)
split_date = data.index[int(N * .8)]
for D in LAGS:
    lagged = pd.DataFrame(index=data.index)
    for col in vars_:
        for lag in range(1, D + 1):
            lagged[f"{col}_lag{lag}"] = data[col].shift(lag)
    for h_name, h in HORIZONS.items():
        target = price.shift(-h)
        ds = lagged.copy()
        ds["target"] = target
        ds = ds.dropna()
        train = ds.loc[ds.index < split_date]
        test = ds.loc[ds.index >= split_date]
        if len(train) < 100 or len(test) < 20:
            continue
        X_train, y_train = train.drop(columns="target"), train["target"]
        X_test, y_test = test.drop(columns="target"), test["target"]
        # RW prediction is current copper price (last observed price), a naive benchmark.
        rw_pred = test["Copper_lag1"].to_numpy()
        for model_name in MODEL_NAMES:
            if model_name == "RW":
                pred = rw_pred
            else:
                model = make_model(model_name)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            results.append({"D": D, "Horizon": h_name, "Horizon_steps": h, "Model": model_name,
                            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                            "MAE": float(mean_absolute_error(y_test, pred)),
                            "Train_n": len(train), "Test_n": len(test), "Test_start": str(test.index.min().date()),
                            "Test_end": str(test.index.max().date())})
    print(f"Finished lag length D={D}")

res = pd.DataFrame(results)
res.to_csv(OUT / "forecast_evaluation_all_models.csv", index=False)
if not res.empty:
    # Table-like pivot: rows horizons, columns model scores, one CSV per lag length
    for metric in ["RMSE", "MAE"]:
        piv = res.pivot_table(index=["D", "Horizon"], columns="Model", values=metric).reindex(columns=MODEL_NAMES)
        piv.to_csv(OUT / f"forecast_table_{metric.lower()}.csv")
    # Table 2/3 style display images, all lag lengths and horizons
    for metric in ["RMSE", "MAE"]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
        for ax, D in zip(axes.ravel(), LAGS):
            subset = res[res["D"] == D]
            p = subset.pivot(index="Horizon", columns="Model", values=metric).reindex(index=list(HORIZONS), columns=MODEL_NAMES)
            p.plot(kind="bar", ax=ax, width=.8)
            ax.set_title(f"D={D}"); ax.set_xlabel(""); ax.set_ylabel(metric); ax.tick_params(axis="x", labelrotation=35); ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"Forecast evaluation by lag length: {metric} (2010–2026; chronological 80/20 split)")
        fig.tight_layout(); fig.savefig(OUT / f"figure4_{metric.lower()}_model_comparison.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    # Compact best-model table per horizon and D
    best = res.loc[res.groupby(["D", "Horizon"])["RMSE"].idxmin()].sort_values(["D", "Horizon"])
    best.to_csv(OUT / "best_model_by_horizon_and_lag.csv", index=False)
else:
    print("No forecast results generated; check sample length and CSV data.")

# 7. Forecast-vs-actual illustration for one reasonable configuration (D=10, 1 month)
D, h = 10, HORIZONS["1 month"]
lagged = pd.DataFrame(index=data.index)
for col in vars_:
    for lag in range(1, D + 1):
        lagged[f"{col}_lag{lag}"] = data[col].shift(lag)
target = price.shift(-h)
ds = lagged.copy(); ds["target"] = target; ds = ds.dropna()
train, test = ds.loc[ds.index < split_date], ds.loc[ds.index >= split_date]
if len(train) > 100 and len(test) > 20:
    model = make_model("RF"); model.fit(train.drop(columns="target"), train["target"])
    pred = model.predict(test.drop(columns="target"))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(test.index, test["target"], label="Actual future copper price", linewidth=1)
    ax.plot(test.index, pred, label="Random Forest forecast", linewidth=1)
    ax.set_title("Figure 5. Actual vs. Random Forest Forecast (D=10, 1-month horizon)")
    ax.set_ylabel("Copper price"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "figure5_actual_vs_forecast.png", dpi=200, bbox_inches="tight"); plt.close(fig)

print(f"\nDone. Tables and figures saved in: {OUT}")
print("Note: horizons use approximate trading-day steps; models and split are a classroom implementation, not exact paper replication.")

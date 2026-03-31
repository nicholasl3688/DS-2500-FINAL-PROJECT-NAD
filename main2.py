import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------------------------------------------------
# LOAD & MERGE
# -----------------------------------------------------------------------
base = "Updated Data/"

hpi    = pd.read_csv(base + "HousingPrices.csv",       parse_dates=["observation_date"])
delinq = pd.read_csv(base + "Delinquency.csv",         parse_dates=["observation_date"])
unemp  = pd.read_csv(base + "UnemploymentRate.csv",    parse_dates=["observation_date"])
conf   = pd.read_csv(base + "ConsumerConfidence.csv",  parse_dates=["observation_date"])

print("COLUMN NAMES")
print("-" * 60)
print(f"  HousingPrices.csv:      {list(hpi.columns)}")
print(f"  Delinquency.csv:        {list(delinq.columns)}")
print(f"  UnemploymentRate.csv:   {list(unemp.columns)}")
print(f"  ConsumerConfidence.csv: {list(conf.columns)}")

df = (hpi.merge(delinq, on="observation_date")
         .merge(unemp,  on="observation_date")
         .merge(conf,   on="observation_date")
         .sort_values("observation_date")
         .reset_index(drop=True))

# -----------------------------------------------------------------------
# TRAIN / TEST SPLIT  (chronological — train < 2014, test >= 2014)
# -----------------------------------------------------------------------
cutoff = pd.Timestamp("2014-01-01")
train = df[df["observation_date"] < cutoff]
test  = df[df["observation_date"] >= cutoff]

features = ["DRSFRMACBS_PCH", "UNRATE_PCH", "USACSCICP02STSAM_PCH"]
target   = "USSTHPI_PCH"

X_train, y_train = train[features].values, train[target].values
X_test,  y_test  = test[features].values,  test[target].values

print("\nHOUSING PRICE INDEX — LINEAR REGRESSION")
print("=" * 60)
print(f"Train: {len(train)} quarters  ({train['observation_date'].min().date()} – {train['observation_date'].max().date()})")
print(f"Test:  {len(test)} quarters  ({test['observation_date'].min().date()} – {test['observation_date'].max().date()})")

# -----------------------------------------------------------------------
# TRAIN MODEL
# -----------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------------------------------------------------
# PREDICTIONS
# -----------------------------------------------------------------------
y_pred_test  = model.predict(X_test)
y_pred_train = model.predict(X_train)

# -----------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------
n, p = X_test.shape

r2       = r2_score(y_test, y_pred_test)
adj_r2   = 1 - (1 - r2) * (n - 1) / (n - p - 1)
mse      = mean_squared_error(y_test, y_pred_test)
rmse     = np.sqrt(mse)
train_r2 = r2_score(y_train, y_pred_train)

print("\nTEST SET RESULTS (2014 Q1 – 2019 Q4)")
print("-" * 60)
print(f"  R²:                   {r2:.4f}")
print(f"  Adjusted R²:          {adj_r2:.4f}")
print(f"  MSE:                  {mse:.4f}")
print(f"  RMSE:                 {rmse:.4f}")
print(f"  Train R² (in-sample): {train_r2:.4f}")

# -----------------------------------------------------------------------
# MODEL COEFFICIENTS
# -----------------------------------------------------------------------
print("\nMODEL COEFFICIENTS")
print("-" * 60)
labels = ["Delinquency Rate %Δ", "Unemployment Rate %Δ", "Consumer Confidence %Δ"]
print(f"  {'Intercept':<30} {model.intercept_:>10.4f}")
for label, coef in zip(labels, model.coef_):
    print(f"  {label:<30} {coef:>10.4f}")

# -----------------------------------------------------------------------
# EXPORT RESULTS TO CSV
# -----------------------------------------------------------------------
output_filename = "predictions_2014_2019.csv"
with open(output_filename, "w") as f:
    f.write("date,predicted_housing_price_pct_change,actual_housing_price_pct_change\n")
    for date, pred, actual in zip(test["observation_date"], y_pred_test, y_test):
        f.write(f"{date.date()},{pred:.6f},{actual:.6f}\n")

print(f"\nResults saved to {output_filename}")

# -----------------------------------------------------------------------
# PLOT — PREDICTED VS ACTUAL
# -----------------------------------------------------------------------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(test["observation_date"], y_test,
        label="Actual", marker="o", linewidth=2)
ax.plot(test["observation_date"], y_pred_test,
        label="Predicted", marker="x", linestyle="--", linewidth=2)

ax.set_title("Housing Price Index % Change: Predicted vs Actual (2014–2019)", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Quarterly % Change")
ax.legend()
ax.annotate(
    f"R² = {r2:.4f}  |  Adj R² = {adj_r2:.4f}  |  MSE = {mse:.4f}",
    xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
)

plt.tight_layout()
plt.show()

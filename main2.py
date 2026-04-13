import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def run_model(delinquency=0, unemployment=0, consumer_conf=0, mortgage=0, nasdaq=0):

    # -----------------------------------------------------------------------
    # LOAD & MERGE
    # -----------------------------------------------------------------------
    base = "Updated Data/"

    hpi    = pd.read_csv(base + "HousingPrices.csv",      parse_dates=["observation_date"])
    delinq = pd.read_csv(base + "Delinquency.csv",         parse_dates=["observation_date"])
    unemp  = pd.read_csv(base + "UnemploymentRate.csv",    parse_dates=["observation_date"])
    conf   = pd.read_csv(base + "ConsumerConfidence.csv",  parse_dates=["observation_date"])
    mort   = pd.read_csv(base + "MortgageRates.csv",       parse_dates=["observation_date"])
    nasd   = pd.read_csv(base + "Nasdaq.csv",              parse_dates=["observation_date"])

    print("COLUMN NAMES")
    print("-" * 60)
    print(f"  HousingPrices.csv:      {list(hpi.columns)}")
    print(f"  Delinquency.csv:        {list(delinq.columns)}")
    print(f"  UnemploymentRate.csv:   {list(unemp.columns)}")
    print(f"  ConsumerConfidence.csv: {list(conf.columns)}")
    print(f"  MortgageRates.csv:      {list(mort.columns)}")
    print(f"  Nasdaq.csv:             {list(nasd.columns)}")

    df = (hpi.merge(delinq, on="observation_date")
             .merge(unemp,  on="observation_date")
             .merge(conf,   on="observation_date")
             .merge(mort,   on="observation_date")
             .merge(nasd,   on="observation_date")
             .sort_values("observation_date")
             .reset_index(drop=True))

    # -----------------------------------------------------------------------
    # APPLY PER-VARIABLE LAGS
    # -----------------------------------------------------------------------
    lag_map = {
        "DRSFRMACBS_PCH":       delinquency,
        "UNRATE_PCH":           unemployment,
        "USACSCICP02STSAM_PCH": consumer_conf,
        "MORTGAGE30US_PCH":     mortgage,
        "NASDAQCOM_PCH":        nasdaq,
    }
    for col, lag in lag_map.items():
        if lag > 0:
            df[col] = df[col].shift(lag)
    df = df.dropna().reset_index(drop=True)

    # -----------------------------------------------------------------------
    # TRAIN / TEST SPLIT  (chronological — train < 2014, test >= 2014)
    # -----------------------------------------------------------------------
    cutoff = pd.Timestamp("2014-01-01")
    train = df[df["observation_date"] < cutoff]
    test  = df[df["observation_date"] >= cutoff]

    features = ["DRSFRMACBS_PCH", "UNRATE_PCH", "USACSCICP02STSAM_PCH", "MORTGAGE30US_PCH", "NASDAQCOM_PCH"]
    target   = "USSTHPI_PCH"

    X_train, y_train = train[features].values, train[target].values
    X_test,  y_test  = test[features].values,  test[target].values

    print("\nHOUSING PRICE INDEX % CHANGE — LINEAR REGRESSION")
    print("=" * 60)
    print(f"  Lags: delinquency={delinquency}, unemployment={unemployment}, "
          f"consumer_conf={consumer_conf}, mortgage={mortgage}, nasdaq={nasdaq}")
    print(f"  Train: {len(train)} quarters  ({train['observation_date'].min().date()} – {train['observation_date'].max().date()})")
    print(f"  Test:  {len(test)} quarters  ({test['observation_date'].min().date()} – {test['observation_date'].max().date()})")

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
    labels = ["Delinquency Rate %Δ", "Unemployment Rate %Δ", "Consumer Confidence %Δ", "Mortgage Rate %Δ", "NASDAQ %Δ"]

    print("\nMODEL COEFFICIENTS")
    print("-" * 60)
    print(f"  {'Intercept':<30} {model.intercept_:>10.4f}")
    for label, coef in zip(labels, model.coef_):
        print(f"  {label:<30} {coef:>10.4f}")

    # -----------------------------------------------------------------------
    # FEATURE WEIGHTS (training set)
    # -----------------------------------------------------------------------
    print("\nFEATURE WEIGHTS (training set)")
    print("-" * 60)
    for label, coef in zip(labels, model.coef_):
        print(f"  {label:<30} {coef:>10.4f}")

    # -----------------------------------------------------------------------
    # PLOT — PREDICTED VS ACTUAL
    # -----------------------------------------------------------------------
    lag_note = f"delinquency={delinquency}, unemployment={unemployment}, consumer_conf={consumer_conf}, mortgage={mortgage}, nasdaq={nasdaq}"

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(test["observation_date"], y_test,
            label="Actual", marker="o", linewidth=2)
    ax.plot(test["observation_date"], y_pred_test,
            label="Predicted", marker="x", linestyle="--", linewidth=2)

    ax.set_title(f"Housing Price Index % Change: Predicted vs Actual (2014–2019)\nLags: {lag_note}", fontsize=11)
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


if __name__ == "__main__":
    run_model()

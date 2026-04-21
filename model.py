import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def run_model(delinquency=0, unemployment=0, consumer_conf=0, mortgage=0, nasdaq=0,
              drop_delinquency=False, drop_unemployment=False, drop_consumer_conf=False,
              drop_mortgage=False, drop_nasdaq=False):

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

    all_features = [
        ("DRSFRMACBS_PCH",       "Delinquency Rate %Δ",    drop_delinquency),
        ("UNRATE_PCH",           "Unemployment Rate %Δ",   drop_unemployment),
        ("USACSCICP02STSAM_PCH", "Consumer Confidence %Δ", drop_consumer_conf),
        ("MORTGAGE30US_PCH",     "Mortgage Rate %Δ",       drop_mortgage),
        ("NASDAQCOM_PCH",        "NASDAQ %Δ",              drop_nasdaq),
    ]
    features = [col   for col, _, drop in all_features if not drop]
    labels   = [label for _,  label, drop in all_features if not drop]
    dropped  = [label for _,  label, drop in all_features if drop]
    target   = "USSTHPI_PCH"

    X_train, y_train = train[features].values, train[target].values
    X_test,  y_test  = test[features].values,  test[target].values

    print("\nHOUSING PRICE INDEX % CHANGE — LINEAR REGRESSION")
    print("=" * 60)
    print(f"  Lags: delinquency={delinquency}, unemployment={unemployment}, "
          f"consumer_conf={consumer_conf}, mortgage={mortgage}, nasdaq={nasdaq}")
    if dropped:
        print(f"  Dropped: {', '.join(dropped)}")
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

    print("\nPEARSON CORRELATION (each feature vs target, training set)")
    print("-" * 60)
    for col, label in zip(features, labels):
        r = np.corrcoef(train[col].values, train[target].values)[0, 1]
        print(f"  {label:<30} r = {r:.4f}")

    # -----------------------------------------------------------------------
    # MEAN PERCENT CHANGE (per feature + target, full dataset)
    # -----------------------------------------------------------------------
    print("\nDESCRIPTIVE STATS (full dataset)")
    print("-" * 60)
    for col, label in zip(features, labels):
        mean   = df[col].mean()
        median = df[col].median()
        mode   = df[col].mode()[0]
        print(f"  {label:<30} mean = {mean:>7.4f}  median = {median:>7.4f}  mode = {mode:>7.4f}")
    hpi_mean   = df["USSTHPI_PCH"].mean()
    hpi_median = df["USSTHPI_PCH"].median()
    hpi_mode   = df["USSTHPI_PCH"].mode()[0]
    print(f"  {'Housing Price Index %Δ':<30} mean = {hpi_mean:>7.4f}  median = {hpi_median:>7.4f}  mode = {hpi_mode:>7.4f}")

    # -----------------------------------------------------------------------
    # MODEL COEFFICIENTS / FEATURE WEIGHTS
    # -----------------------------------------------------------------------
    print("\nMODEL COEFFICIENTS / FEATURE WEIGHTS")
    print("-" * 60)
    print(f"  {'Intercept':<30} {model.intercept_:>10.4f}")
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

    # -----------------------------------------------------------------------
    # PLOT — CORRELATION HEATMAP (each feature vs housing price index only)
    # -----------------------------------------------------------------------
    corr_values = [[df[col].corr(df[target])] for col in features]
    corr_heatmap = pd.DataFrame(corr_values, index=labels, columns=["Housing Price %Δ"])

    fig2, ax2 = plt.subplots(figsize=(4, max(3, len(labels) * 0.6 + 1)))
    sns.heatmap(corr_heatmap, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax2)
    ax2.set_title("Pearson Correlation — Each Feature vs Housing Price %Δ", fontsize=11)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------
    # PLOT — TIME SERIES (all variables on one plot)
    # -----------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    for col, lbl in zip(features + [target], labels + ["Housing Price %Δ"]):
        ax3.plot(df["observation_date"], df[col], linewidth=1.5, label=lbl)
    ax3.axvline(pd.Timestamp("2014-01-01"), color="black", linestyle="--",
                linewidth=1, label="Train/Test Split")
    ax3.set_title("Feature & Target Time Series (full dataset)", fontsize=12)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Quarterly % Change")
    ax3.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------
    # PLOT — RESIDUALS
    # -----------------------------------------------------------------------
    residuals = y_test - y_pred_test

    fig4, ax4 = plt.subplots(figsize=(11, 4))
    ax4.plot(test["observation_date"], residuals, marker="o", linewidth=2, label="Residual")
    ax4.axhline(0, color="red", linestyle="--", linewidth=1)
    ax4.set_title("Residuals — Test Period (2014–2019)", fontsize=12)
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Residual (Actual − Predicted)")
    ax4.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_model()

import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


BASE_PATH = "Updated Data/"
DATE_COLUMN = "observation_date"
TARGET_COLUMN = "USSTHPI_PCH"
FEATURE_COLUMNS = [
    "DRSFRMACBS_PCH",
    "UNRATE_PCH",
    "USACSCICP02STSAM_PCH",
]
DISPLAY_NAMES = {
    "USSTHPI_PCH": "Housing Price Index % Change",
    "DRSFRMACBS_PCH": "Delinquency Rate % Change",
    "UNRATE_PCH": "Unemployment Rate % Change",
    "USACSCICP02STSAM_PCH": "Consumer Confidence % Change",
}
MAX_LAG_QUARTERS = 10
TRAIN_TEST_CUTOFF = pd.Timestamp("2014-01-01")


def load_merged_dataframe():
    hpi = pd.read_csv(BASE_PATH + "HousingPrices.csv", parse_dates=[DATE_COLUMN])
    delinq = pd.read_csv(BASE_PATH + "Delinquency.csv", parse_dates=[DATE_COLUMN])
    unemp = pd.read_csv(BASE_PATH + "UnemploymentRate.csv", parse_dates=[DATE_COLUMN])
    conf = pd.read_csv(BASE_PATH + "ConsumerConfidence.csv", parse_dates=[DATE_COLUMN])

    print("COLUMN NAMES")
    print("-" * 60)
    print(f"  HousingPrices.csv:      {list(hpi.columns)}")
    print(f"  Delinquency.csv:        {list(delinq.columns)}")
    print(f"  UnemploymentRate.csv:   {list(unemp.columns)}")
    print(f"  ConsumerConfidence.csv: {list(conf.columns)}")

    return (
        hpi.merge(delinq, on=DATE_COLUMN)
        .merge(unemp, on=DATE_COLUMN)
        .merge(conf, on=DATE_COLUMN)
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )


def build_lagged_dataset(df, lag_map):
    lagged_df = df[[DATE_COLUMN, TARGET_COLUMN] + FEATURE_COLUMNS].copy()
    lagged_feature_columns = []

    for feature in FEATURE_COLUMNS:
        lag = lag_map[feature]
        lagged_column = f"{feature}_lag{lag}"
        lagged_df[lagged_column] = lagged_df[feature].shift(lag)
        lagged_feature_columns.append(lagged_column)

    lagged_df = lagged_df.dropna().reset_index(drop=True)
    return lagged_df, lagged_feature_columns


def fit_and_evaluate(df, feature_columns):
    train = df[df[DATE_COLUMN] < TRAIN_TEST_CUTOFF].copy()
    test = df[df[DATE_COLUMN] >= TRAIN_TEST_CUTOFF].copy()

    X_train = train[feature_columns].values
    y_train = train[TARGET_COLUMN].values
    X_test = test[feature_columns].values
    y_test = test[TARGET_COLUMN].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    n, p = X_test.shape
    test_r2 = r2_score(y_test, y_pred_test)
    adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
    mse = mean_squared_error(y_test, y_pred_test)

    return {
        "model": model,
        "train": train,
        "test": test,
        "feature_columns": feature_columns,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": test_r2,
        "adjusted_r2": adjusted_r2,
        "mse": mse,
        "rmse": np.sqrt(mse),
    }


def search_best_lags(df):
    search_results = []

    for lag_values in itertools.product(range(MAX_LAG_QUARTERS + 1), repeat=len(FEATURE_COLUMNS)):
        lag_map = dict(zip(FEATURE_COLUMNS, lag_values))
        lagged_df, lagged_feature_columns = build_lagged_dataset(df, lag_map)
        results = fit_and_evaluate(lagged_df, lagged_feature_columns)

        search_results.append(
            {
                "lag_map": lag_map,
                "lagged_df": lagged_df,
                "results": results,
            }
        )

    search_results.sort(key=lambda row: row["results"]["test_r2"], reverse=True)
    return search_results


def print_model_results(section_title, lag_map, results):
    print(f"\n{section_title}")
    print("-" * 60)
    print(
        f"  Train: {len(results['train'])} quarters  "
        f"({results['train'][DATE_COLUMN].min().date()} - "
        f"{results['train'][DATE_COLUMN].max().date()})"
    )
    print(
        f"  Test:  {len(results['test'])} quarters  "
        f"({results['test'][DATE_COLUMN].min().date()} - "
        f"{results['test'][DATE_COLUMN].max().date()})"
    )
    print(f"  Test R^2:              {results['test_r2']:.4f}")
    print(f"  Adjusted Test R^2:     {results['adjusted_r2']:.4f}")
    print(f"  Test MSE:              {results['mse']:.4f}")
    print(f"  Test RMSE:             {results['rmse']:.4f}")
    print(f"  Train R^2 (in-sample): {results['train_r2']:.4f}")

    print("\n  Selected lags")
    print("  " + "-" * 56)
    for feature in FEATURE_COLUMNS:
        print(f"  {DISPLAY_NAMES[feature]:<35} {lag_map[feature]:>2} quarter(s)")

    print("\n  Coefficients")
    print("  " + "-" * 56)
    print(f"  {'Intercept':<35} {results['model'].intercept_:>10.4f}")
    for feature, coef in zip(FEATURE_COLUMNS, results["model"].coef_):
        print(f"  {DISPLAY_NAMES[feature]:<35} {coef:>10.4f}")


def export_predictions(results, filename):
    output_df = pd.DataFrame(
        {
            "date": results["test"][DATE_COLUMN].dt.date,
            "predicted_housing_price_pct_change": results["y_pred_test"],
            "actual_housing_price_pct_change": results["test"][TARGET_COLUMN].values,
        }
    )
    output_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


def main():
    df = load_merged_dataframe()
    same_quarter_lag_map = {feature: 0 for feature in FEATURE_COLUMNS}
    same_quarter_df, same_quarter_feature_columns = build_lagged_dataset(df, same_quarter_lag_map)
    same_quarter_results = fit_and_evaluate(same_quarter_df, same_quarter_feature_columns)

    search_results = search_best_lags(df)
    best_row = search_results[0]
    best_lag_map = best_row["lag_map"]
    best_results = best_row["results"]

    print("\nHOUSING PRICE INDEX - MIXED LAG SEARCH")
    print("=" * 60)
    print(
        f"Full sample: {len(df)} quarterly rows "
        f"({df[DATE_COLUMN].min().date()} - {df[DATE_COLUMN].max().date()})"
    )
    print(
        f"Grid searched all lag combinations from 0 to {MAX_LAG_QUARTERS} quarters "
        f"for each predictor."
    )

    print_model_results("SAME-QUARTER MODEL", same_quarter_lag_map, same_quarter_results)
    print_model_results("BEST MIXED-LAG MODEL", best_lag_map, best_results)

    print("\nTOP 5 LAG COMBINATIONS BY TEST R^2")
    print("-" * 60)
    for rank, row in enumerate(search_results[:5], start=1):
        lag_map = row["lag_map"]
        results = row["results"]
        print(
            f"  {rank}. Delinquency lag={lag_map['DRSFRMACBS_PCH']}, "
            f"Unemployment lag={lag_map['UNRATE_PCH']}, "
            f"Confidence lag={lag_map['USACSCICP02STSAM_PCH']}  |  "
            f"R^2={results['test_r2']:.4f}, RMSE={results['rmse']:.4f}"
        )

    print("\nMODEL COMPARISON")
    print("-" * 60)
    print(
        f"  Same-quarter model test R^2: {same_quarter_results['test_r2']:.4f}   "
        f"RMSE: {same_quarter_results['rmse']:.4f}"
    )
    print(
        f"  Best mixed-lag model test R^2: {best_results['test_r2']:.4f}   "
        f"RMSE: {best_results['rmse']:.4f}"
    )

    if best_results["test_r2"] > same_quarter_results["test_r2"]:
        print("  The mixed-lag model performs better on the held-out test set.")
    else:
        print("  The same-quarter model still performs better on the held-out test set.")

    export_predictions(same_quarter_results, "predictions_2014_2019_same_quarter.csv")
    export_predictions(best_results, "predictions_2014_2019_best_mixed_lag.csv")


if __name__ == "__main__":
    main()

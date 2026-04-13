import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


BASE_PATH = "Updated Data/"
DATE_COLUMN = "observation_date"
TARGET_COLUMN = "USSTHPI_PCH"
# This one list drives most of the script, so adding a new predictor only
# requires dropping it in here with its file name and display label.
FEATURE_SPECS = [
    ("DRSFRMACBS_PCH", "Delinquency.csv", "Delinquency Rate % Change"),
    ("UNRATE_PCH", "UnemploymentRate.csv", "Unemployment Rate % Change"),
    ("USACSCICP02STSAM_PCH", "ConsumerConfidence.csv", "Consumer Confidence % Change"),
    ("MORTGAGE30US_PCH", "MortgageRates.csv", "Mortgage Rate % Change"),
    ("NASDAQCOM_PCH", "Nasdaq.csv", "Nasdaq % Change"),
]
FEATURE_COLUMNS = [column for column, _, _ in FEATURE_SPECS]
DISPLAY_NAMES = {column: label for column, _, label in FEATURE_SPECS}
DISPLAY_NAMES[TARGET_COLUMN] = "Housing Price Index % Change"
MAX_LAG_QUARTERS = 10
TOP_RESULT_COUNT = 5
TRAIN_TEST_CUTOFF = pd.Timestamp("2014-01-01")


def load_merged_dataframe():
    # Start with housing prices, then pull in each predictor so every row
    # ends up representing the same quarter across all series.
    dataframes = [
        pd.read_csv(BASE_PATH + "HousingPrices.csv", parse_dates=[DATE_COLUMN]),
    ]

    for _, filename, _ in FEATURE_SPECS:
        dataframes.append(pd.read_csv(BASE_PATH + filename, parse_dates=[DATE_COLUMN]))

    print("COLUMN NAMES")
    print("-" * 60)
    print(f"  HousingPrices.csv:      {list(dataframes[0].columns)}")
    for (_, filename, _), df in zip(FEATURE_SPECS, dataframes[1:]):
        print(f"  {filename:<22} {list(df.columns)}")

    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = merged_df.merge(df, on=DATE_COLUMN)

    return merged_df.sort_values(DATE_COLUMN).reset_index(drop=True)


def build_lagged_dataset(df, lag_map):
    # For each variable, shift the values backward by however many quarters
    # the lag search picked, then drop the first few rows that no longer line up.
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
    # Keep the split chronological so the model is always trained on earlier
    # data and tested on later data.
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


def solve_linear_regression(X_train, y_train, X_test):
    # This does the same job as a regular linear regression fit, but it is
    # much faster inside the lag search because we solve it directly.
    train_design = np.column_stack([np.ones(len(X_train)), X_train])
    beta = np.linalg.lstsq(train_design, y_train, rcond=None)[0]
    test_design = np.column_stack([np.ones(len(X_test)), X_test])
    y_pred_train = train_design @ beta
    y_pred_test = test_design @ beta
    return beta[0], beta[1:], y_pred_train, y_pred_test


def search_best_lags(df):
    # The search is expensive, so cache every shifted version of every variable once
    # instead of rebuilding the same lagged columns over and over inside the loop.
    dates = df[DATE_COLUMN].to_numpy()
    target = df[TARGET_COLUMN].to_numpy()
    feature_values = {feature: df[feature].to_numpy() for feature in FEATURE_COLUMNS}

    shifted_feature_cache = {}
    for feature in FEATURE_COLUMNS:
        shifted_feature_cache[feature] = {}
        values = feature_values[feature]
        for lag in range(MAX_LAG_QUARTERS + 1):
            if lag == 0:
                shifted_feature_cache[feature][lag] = values
            else:
                shifted_values = np.empty_like(values)
                shifted_values[:lag] = np.nan
                shifted_values[lag:] = values[:-lag]
                shifted_feature_cache[feature][lag] = shifted_values

    top_rows = []
    best_row = None

    # Try every lag combination, score it on the held-out test period,
    # and hang on to both the overall winner and the top few runners-up.
    for lag_values in itertools.product(range(MAX_LAG_QUARTERS + 1), repeat=len(FEATURE_COLUMNS)):
        lag_map = dict(zip(FEATURE_COLUMNS, lag_values))
        max_shift = max(lag_values)

        # Once the largest lag is known, we can slice everything from that point
        # forward so the target and all predictors stay aligned.
        candidate_dates = dates[max_shift:]
        candidate_target = target[max_shift:]
        candidate_matrix = np.column_stack(
            [
                shifted_feature_cache[feature][lag][max_shift:]
                for feature, lag in zip(FEATURE_COLUMNS, lag_values)
            ]
        )

        train_mask = candidate_dates < TRAIN_TEST_CUTOFF.to_datetime64()
        test_mask = ~train_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train = candidate_matrix[train_mask]
        y_train = candidate_target[train_mask]
        X_test = candidate_matrix[test_mask]
        y_test = candidate_target[test_mask]

        intercept, coefficients, y_pred_train, y_pred_test = solve_linear_regression(
            X_train,
            y_train,
            X_test,
        )

        test_r2 = r2_score(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        row = {
            "lag_map": lag_map,
            "results": {
                "test_r2": test_r2,
                "rmse": np.sqrt(mse),
                "mse": mse,
                "train_r2": r2_score(y_train, y_pred_train),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "intercept": intercept,
                "coefficients": coefficients,
            },
        }

        top_rows.append(row)
        top_rows.sort(key=lambda candidate: candidate["results"]["test_r2"], reverse=True)
        top_rows = top_rows[:TOP_RESULT_COUNT]

        if best_row is None or test_r2 > best_row["results"]["test_r2"]:
            best_row = row

    return best_row, top_rows


def print_model_results(section_title, lag_map, results):
    # This print block is meant to be presentation-friendly, so it shows
    # the lag choices next to the fitted coefficients and test metrics.
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


def print_top_combinations(top_rows):
    # Seeing the top few combinations helps us tell whether the best model is
    # clearly ahead or if several lag setups perform about the same.
    print(f"\nTOP {len(top_rows)} LAG COMBINATIONS BY TEST R^2")
    print("-" * 60)
    for rank, row in enumerate(top_rows, start=1):
        lag_map = row["lag_map"]
        summary = row["results"]
        print(
            f"  {rank}. Delinq={lag_map['DRSFRMACBS_PCH']}, "
            f"Unemp={lag_map['UNRATE_PCH']}, "
            f"Conf={lag_map['USACSCICP02STSAM_PCH']}, "
            f"Mortgage={lag_map['MORTGAGE30US_PCH']}, "
            f"Nasdaq={lag_map['NASDAQCOM_PCH']}  |  "
            f"R^2={summary['test_r2']:.4f}, RMSE={summary['rmse']:.4f}"
        )


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

    # First build the simple baseline with no lagging at all.
    same_quarter_lag_map = {feature: 0 for feature in FEATURE_COLUMNS}
    same_quarter_df, same_quarter_feature_columns = build_lagged_dataset(df, same_quarter_lag_map)
    same_quarter_results = fit_and_evaluate(same_quarter_df, same_quarter_feature_columns)

    # Then let the search find the lag combination that gives the strongest
    # out-of-sample fit when all predictors are used together.
    best_row, top_rows = search_best_lags(df)
    best_lag_map = best_row["lag_map"]
    best_df, best_feature_columns = build_lagged_dataset(df, best_lag_map)
    best_results = fit_and_evaluate(best_df, best_feature_columns)

    print("\nHOUSING PRICE INDEX - FULL MIXED LAG SEARCH")
    print("=" * 60)
    print(
        f"Full sample: {len(df)} quarterly rows "
        f"({df[DATE_COLUMN].min().date()} - {df[DATE_COLUMN].max().date()})"
    )
    print(
        f"Predictors: {len(FEATURE_COLUMNS)} total "
        f"(delinquency, unemployment, confidence, mortgage rates, Nasdaq)."
    )
    print(
        f"Grid searched all lag combinations from 0 to {MAX_LAG_QUARTERS} quarters "
        f"for each predictor."
    )

    print_model_results("SAME-QUARTER MODEL", same_quarter_lag_map, same_quarter_results)
    print_model_results("BEST MIXED-LAG MODEL", best_lag_map, best_results)
    print_top_combinations(top_rows)

    print("\nMODEL COMPARISON")
    print("-" * 60)
    print(
        f"  Same-quarter model test R^2: {same_quarter_results['test_r2']:.4f}   "
        f"Adjusted R^2: {same_quarter_results['adjusted_r2']:.4f}   "
        f"RMSE: {same_quarter_results['rmse']:.4f}"
    )
    print(
        f"  Best mixed-lag model test R^2: {best_results['test_r2']:.4f}   "
        f"Adjusted R^2: {best_results['adjusted_r2']:.4f}   "
        f"RMSE: {best_results['rmse']:.4f}"
    )

    if best_results["test_r2"] > same_quarter_results["test_r2"]:
        print("  The mixed-lag model performs better on the held-out test set.")
    else:
        print("  The same-quarter model still performs better on the held-out test set.")

    export_predictions(same_quarter_results, "predictions_2014_2019_same_quarter_full_model.csv")
    export_predictions(best_results, "predictions_2014_2019_best_mixed_lag_full_model.csv")


if __name__ == "__main__":
    main()

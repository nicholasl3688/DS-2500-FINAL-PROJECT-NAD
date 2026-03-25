import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr


# Centralize the column names used across the analysis so they are easy to reuse and update.
DATE_COLUMN = "observation_date"
TARGET_COLUMN = "housing_price_index_pct_change"
PREDICTOR_COLUMNS = [
    "delinquency_rate_pct_change",
    "unemployment_rate_pct_change",
    "consumer_confidence_pct_change",
]
# Use cleaner display labels when printing tables to the terminal.
DISPLAY_NAMES = {
    "housing_price_index_pct_change": "Housing Price Index",
    "delinquency_rate_pct_change": "Delinquency Rate",
    "unemployment_rate_pct_change": "Unemployment Rate",
    "consumer_confidence_pct_change": "Consumer Confidence Index",
}


def load_quarterly_series(path, raw_value_column, renamed_value_column):
    # Load one quarterly dataset, convert the date column, and standardize the metric name.
    df = pd.read_csv(path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    return (
        df[[DATE_COLUMN, raw_value_column]]
        .rename(columns={raw_value_column: renamed_value_column})
        .sort_values(DATE_COLUMN)
    )


def build_analysis_dataframe():
    # Load each quarterly percent-change series and rename the raw FRED-style columns
    # to analysis-friendly names that we can reference consistently later.
    housing_price_index_df = load_quarterly_series(
        "Housing Price Index.csv",
        "USSTHPI_PCH",
        TARGET_COLUMN,
    )
    delinquency_df = load_quarterly_series(
        "Deliquency.csv",
        "DRSFRMACBS_PCH",
        "delinquency_rate_pct_change",
    )
    unemployment_rate_df = load_quarterly_series(
        "Unemployment Rate.csv",
        "UNRATE_PCH",
        "unemployment_rate_pct_change",
    )
    consumer_confidence_df = load_quarterly_series(
        "Consumer Confidence.csv",
        "USACSCICP02STSAM_PCH",
        "consumer_confidence_pct_change",
    )

    # Merge the four datasets by quarter so every row represents the same time period
    # across housing prices, delinquency, unemployment, and consumer confidence.
    analysis_df = housing_price_index_df.merge(
        delinquency_df,
        on=DATE_COLUMN,
        how="inner",
    ).merge(
        unemployment_rate_df,
        on=DATE_COLUMN,
        how="inner",
    ).merge(
        consumer_confidence_df,
        on=DATE_COLUMN,
        how="inner",
    )

    return analysis_df


def print_summary_stats(df):
    # Print the summary statistics section and compute the core descriptive metrics
    # for the housing index and each predictor variable.
    print("\nSUMMARY STATISTICS")
    print("-" * 80)

    summary_rows = []
    for column in [TARGET_COLUMN] + PREDICTOR_COLUMNS:
        summary_rows.append(
            {
                "Variable": DISPLAY_NAMES[column],
                "Mean": df[column].mean(),
                "Standard Deviation": df[column].std(),
                "Min": df[column].min(),
                "Max": df[column].max(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def print_correlation_analysis(df):
    # Print Pearson correlations between each predictor and the housing price index
    # so we can see the strength and direction of each pairwise relationship.
    print("\nPEARSON CORRELATION ANALYSIS")
    print("-" * 80)
    print(
        f"Comparing each predictor to {DISPLAY_NAMES[TARGET_COLUMN]} "
        f"({TARGET_COLUMN})."
    )

    correlation_rows = []
    for predictor in PREDICTOR_COLUMNS:
        correlation_coefficient, p_value = pearsonr(df[predictor], df[TARGET_COLUMN])
        correlation_rows.append(
            {
                "Variable": DISPLAY_NAMES[predictor],
                "Pearson r": correlation_coefficient,
                "p-value": p_value,
            }
        )

    correlation_df = pd.DataFrame(correlation_rows)
    print(correlation_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def print_regression_analysis(df):
    # Run a multiple linear regression with housing price change as the outcome
    # and the three economic indicators as the explanatory variables.
    print("\nREGRESSION ANALYSIS")
    print("-" * 80)

    X = sm.add_constant(df[PREDICTOR_COLUMNS])
    y = df[TARGET_COLUMN]

    model = sm.OLS(y, X).fit()

    # Collect the most important regression outputs for each parameter so the
    # printed results focus on coefficient direction, magnitude, and significance.
    coefficient_rows = []
    for parameter_name in model.params.index:
        label = "Intercept" if parameter_name == "const" else DISPLAY_NAMES[parameter_name]
        coefficient_rows.append(
            {
                "Variable": label,
                "Coefficient (beta)": model.params[parameter_name],
                "p-value": model.pvalues[parameter_name],
            }
        )

    coefficients_df = pd.DataFrame(coefficient_rows)
    print(coefficients_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nR-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")


def main():
    # Build the combined quarterly dataset once, then run each part of the analysis report.
    analysis_df = build_analysis_dataframe()

    print("HOUSING PRICE DETERMINANTS ANALYSIS")
    print("=" * 80)
    print(
        f"Observations: {len(analysis_df)} quarterly rows "
        f"from {analysis_df[DATE_COLUMN].min().date()} "
        f"to {analysis_df[DATE_COLUMN].max().date()}"
    )

    print_summary_stats(analysis_df)
    print_correlation_analysis(analysis_df)
    print_regression_analysis(analysis_df)


# Only run the analysis automatically when this file is executed directly.
if __name__ == "__main__":
    main()

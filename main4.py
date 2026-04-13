from main2 import run_model

# Call run_model() with per-variable lags (in quarters).
# Default for each variable is 0 (no lag).
# Example: delinquency=2 means use delinquency rate from 2 quarters ago.

run_model(delinquency=2, mortgage=1)

#delinquency 10
#unemployment 7
#consumer_conf 2
#mortage rate 0
#nasdaq% 4
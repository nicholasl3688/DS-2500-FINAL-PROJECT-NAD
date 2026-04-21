from model import run_model

# Call run_model() with per-variable lags (in quarters).
# Default for each variable is 0 (no lag).
# Example: delinquency=2 means use delinquency rate from 2 quarters ago.

#ORIGINAL MODEL
# run_model(drop_mortgage=True, drop_nasdaq=True)

#model 2
# run_model()

#FOR REGULAR R^2
# run_model(delinquency=10, unemployment=7, consumer_conf=2, mortgage=0, nasdaq=4,
#               drop_delinquency=False, drop_unemployment=False, drop_consumer_conf=False,
#               drop_mortgage=False, drop_nasdaq=False)

# # FOR ADJUSTED R^2:
run_model(delinquency=10, consumer_conf=2, mortgage=0,
              drop_delinquency=False, drop_unemployment=True, drop_consumer_conf=False,
              drop_mortgage=False, drop_nasdaq=True)

#delinquency 10
#unemployment DROP
#consumer_conf 2
#mortage rate 0
#nasdaq% DROP
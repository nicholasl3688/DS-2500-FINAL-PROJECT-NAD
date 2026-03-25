import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

consumer_confidence_df = pd.read_csv("Consumer Confidence.csv")
delinquency_df = pd.read_csv("Deliquency.csv")
housing_price_index_df = pd.read_csv("Housing Price Index.csv")
unemployment_rate_df = pd.read_csv("Unemployment Rate.csv")
zillow_housing_df = pd.read_csv("Zillow Housing Data.csv")


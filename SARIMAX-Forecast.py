import warnings
import yfinance as yf
import pandas as pd
import itertools
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Fetch daily data for EUR/USD
data = yf.download('EURUSD=X', start='2020-01-01', end='2023-06-30')

# We will use 'Close' prices for our analysis
df = pd.DataFrame(data['Close'].dropna())

# Define the p, d, and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q, and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q, and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# Best model parameters initialization
best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None

# Search for the best model parameters
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            temp_model = SARIMAX(df['Close'],
                                 order=param,
                                 seasonal_order=param_seasonal,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)
            results = temp_model.fit()

            # Compare the AIC values
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal

        except:
            continue

print(f'Best SARIMAX model: SARIMAX({best_pdq}x{best_seasonal_pdq}) with AIC={best_aic}')

# Run a rolling forecast for a longer period
n_steps = 365

# Create a new DataFrame to hold predictions
df_predict = pd.DataFrame(index=df.index)

for i in range(len(df)-n_steps, len(df)):
    train_data = df['Close'][:i]
    model = SARIMAX(train_data,
                    order=best_pdq,
                    seasonal_order=best_seasonal_pdq,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=1)
    df_predict.loc[train_data.index[-1], 'predict'] = forecast.predicted_mean[0]

# Combine the original DataFrame with the forecast series
df_combined = pd.concat([df, df_predict], axis=1)

# Compute and plot ACF of residuals
residuals = model_fit.resid
ljung_box_results = acorr_ljungbox(residuals, lags=[10], return_df=True)

# Calculate the error between the forecasted values and the actual close prices
df_combined['error'] = df_combined['Close'] - df_combined['predict']

# Calculate and print the mean absolute percentage error (MAPE)
mape = np.mean(np.abs(df_combined['error'] / df_combined['Close'])) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# Calculate the cumulative sum of the forecast errors
cumulative_errors = df_combined['error'].cumsum()

# Calculate the autocorrelation of the forecast errors
error_autocorrelation = residuals.autocorr()

# Create a subplot figure
fig, ax = plt.subplots(4, 1, figsize=(10, 28))

# Plot a zoomed in version
df_combined[['Close', 'predict']].shift(-1).loc[df_combined.index[-n_steps*2:]].plot(ax=ax[0])  # Shift the close prices one day forward
ax[0].set_title('Zoomed in Close prices and Predictions')

# Plot the autocorrelation of the forecast errors
plot_acf(residuals, lags=30, ax=ax[1])
ax[1].set_title('Autocorrelation of Forecast Errors')

# Plot the cumulative sum of forecast errors
cumulative_errors.plot(ax=ax[2])
ax[2].set_title('Cumulative Sum of Forecast Errors')

# Plot the histogram of forecast errors
df_combined['error'].plot.hist(ax=ax[3], bins=20)
ax[3].set_title('Histogram of Forecast Errors')

plt.tight_layout()
plt.show()

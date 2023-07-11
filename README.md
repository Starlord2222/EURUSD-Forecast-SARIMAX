# SARIMAX Model for EUR/USD Currency Pair Forecasting

This repository contains a Python script for time series forecasting of the EUR/USD currency pair. It uses the SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) model from the `statsmodels` library to perform the analysis.

## Project Description

The main script fetches the daily data for the EUR/USD currency pair from the beginning of 2020 until mid-2023 using the `yfinance` library. The 'Close' prices are extracted for the analysis.

The script uses a SARIMAX model to forecast the 'Close' prices. The SARIMAX model parameters (p, d, q) are optimized by trying different combinations within a specified range and choosing the combination that results in the lowest Akaike Information Criterion (AIC) score.

The script also calculates the residuals of the model, performs a Ljung-Box test, and calculates the Mean Absolute Percentage Error (MAPE) to evaluate the model's performance. 

Finally, it plots several diagrams, including:
- A zoomed-in version of the close prices and predictions
- The autocorrelation of the forecast errors
- The cumulative sum of forecast errors
- The histogram of forecast errors

## Usage

To use this script, you will need Python installed along with the following Python libraries:
- `warnings`
- `yfinance`
- `pandas`
- `itertools`
- `numpy`
- `statsmodels`
- `matplotlib`

To run the script, simply download the `.py` file and run it using a Python interpreter.

## Results

Please note that the results are dependent on the data and the chosen model parameters, and the optimal model parameters may change if the data is updated.

This script is intended as a starting point for time series analysis of financial data and should be adjusted according to the specific needs and constraints of your project.



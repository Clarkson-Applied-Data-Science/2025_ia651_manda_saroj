# 2025_ia651_manda_saroj

IA651 Final Project

📈 Bitcoin Price Prediction using ARIMA
This project aims to forecast weekly Bitcoin prices using time series analysis with the ARIMA model. The dataset contains historical Bitcoin price data, and the model helps us understand trends and make future predictions.

🧠 Project Motivation
Bitcoin, being a volatile and widely discussed cryptocurrency, presents a perfect use case for financial forecasting. Accurately predicting its price can provide valuable insights for investors, analysts, and researchers. This project uses historical price data and applies an ARIMA model to understand patterns and predict future prices.

📁 Dataset
Source: Bitcoin Historical Data CSV

Attributes Used:

Date – Timestamp of the record

Price – Daily closing price of Bitcoin

Vol. – Volume of Bitcoin traded

Change % – Daily percentage change

🔧 The dataset was cleaned by:

Converting Date to datetime format

Removing currency symbols from Price, Vol., and Change %

Converting cleaned columns to numeric types

Setting Date as the index for time series modeling

🧪 Methodology
Step 1: Data cleaning and preprocessing

Step 2: Exploratory data analysis

Step 3: Time series modeling using ARIMA

Step 4: Forecasting future Bitcoin prices

Step 5: Evaluating model performance using RMSE

📊 Model Used
ARIMA (AutoRegressive Integrated Moving Average)

Model order: (p=5, d=1, q=0)

The model was trained on 80% of the data and tested on the remaining 20%.

📉 Evaluation
Metric: Root Mean Squared Error (RMSE)

Visual comparison of actual vs predicted prices shows the model's effectiveness.

🛠️ Libraries Used
pandas

numpy

matplotlib

statsmodels

sklearn

📷 Output Sample
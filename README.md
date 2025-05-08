# 2025_ia651_manda_saroj

IA651 Final Project

# Bitcoin Price Prediction Project
## Overview
This project involves predicting Bitcoin prices based on historical data using statistical and machine learning
techniques. The dataset used in the project contains several features related to Bitcoin&#39;s historical trading
data, including the price, volume, and percentage change for each day. The analysis is aimed at
understanding the patterns and trends in Bitcoin prices over time.

Cryptocurrencies like Bitcoin have demonstrated high volatility and complex price behavior, making them challenging yet valuable targets for time series forecasting. This project focuses on predicting Bitcoin's closing price using historical data and time series modeling techniques. 
The dataset includes key financial indicators such as daily opening and closing prices, trading volume, highest and lowest prices, and percentage change. By applying ARIMA modeling, the project explores underlying patterns, seasonality, and dependencies in Bitcoin's historical performance to generate forecasts. 

In future phases, we aim to expand the model by incorporating external market indicators such as Ethereum prices and Dow Jones indices. This will enable us to build hybrid models (e.g., ARIMA + LSTM) to improve prediction accuracy and provide more robust market insights.

## Dataset
Source: [Bitcoin Historical Data CSV](https://www.investing.com/crypto/bitcoin/historical-data)
The dataset used for this project contains the following columns:

Date: The date of the record.

Price: The closing price of Bitcoin on that date.

Open: The opening price of Bitcoin on that day.

High: The highest price of Bitcoin during that day.

Low: The lowest price of Bitcoin during that day.

Vol.: The trading volume for that day.

Change %: The percentage change in the price of Bitcoin compared to the previous day.

The data is cleaned and transformed for analysis, ensuring all columns are properly formatted and free
from non-numeric characters

#### Data set format
![image](https://github.com/user-attachments/assets/c023cd6d-73a5-42f3-bbf8-9defba0b4829)

## Project Objective
Bitcoin, being a volatile and widely discussed cryptocurrency, presents a perfect use case for financial
forecasting. Accurately predicting its price can provide valuable insights for investors, analysts, and
researchers. This project uses historical price data and applies an ARIMA model to understand patterns and
predict future prices.

## Methodology


### Data Cleaning and Preprocessing

To ensure the dataset was ready for modeling, several cleaning steps were implemented:

1. **Date Parsing**  
   The 'Date' column was converted to datetime format using `pd.to_datetime()`, with coercion enabled to handle any formatting inconsistencies.

2. **Removing Non-Numeric Characters**  
   Columns like 'Price', 'Vol.', and 'Change %' contained symbols and shorthand notations such as `$`, `,`, `%`, `K` (thousands), `M` (millions), and `B` (billions). These were stripped or converted using regex-based replacement to standard numeric formats (e.g., `'K' → e3`, `'M' → e6`).

3. **Numeric Conversion**  
   After cleaning, all target columns were converted to numeric types using `pd.to_numeric()`. This step ensured compatibility with statistical models like ARIMA, which require numerical inputs.

4. **Missing Value Handling**  
   The dataset was checked for `NaN` values introduced during coercion or conversion. Missing rows were removed using `.dropna()` to ensure model training integrity.

5. **Validation**  
   After preprocessing, we verified the data structure and confirmed that the key columns were free from formatting issues or missing data. A `df.isnull().sum()` check showed that all key numeric fields were fully cleaned.

> Overall, this preprocessing step was essential to ensure the dataset was reliable, consistent, and suitable for time series forecasting models.


### Exploratory data analysis.

Exploratory Data Analysis (EDA) is an essential step to understand the underlying structure, patterns, and
relationships in the dataset. It involves both statistical and graphical analysis of the data to uncover
insights. The EDA process in this project includes:

Summary Statistics: Descriptive statistics (like mean, median, min, max, etc.) are calculated to get an initial
understanding of the data.

Distribution of Features: Visualizing the distribution of key variables (like Price, Vol., and Change %) is
helpful to understand their spread and identify potential outliers.

Correlation Analysis: A correlation matrix helps identify relationships between different numerical features.
For instance, it is useful to see if there&#39;s a correlation between Volume and Price, which could provide
insights into market behavior.

### Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to understand the distribution, patterns, and relationships in the dataset. The key steps and insights include:

#### 1. Summary Statistics
We calculated measures such as **mean**, **median**, **standard deviation**, and **range** for all numeric columns. This helped us understand central tendencies and variation across price, volume, and daily change percentages.

#### 2. Distribution and Outliers (Box Plots)
Box plots were created for `Price`, `Volume`, and `Change %` to visualize the distribution and identify outliers.

- **Price:** The price distribution is right-skewed with several extreme values above $80,000.
  
![Box Plot - Price](https://github.com/user-attachments/assets/c2f88e2d-f5e6-4b41-9f26-7f285e808df4)





- **Volume:** Trading volume showed a heavy concentration near zero with many large outliers, suggesting occasional spikes in trading activity.

  ![Box Plot - Volume](https://github.com/user-attachments/assets/34bd307e-fc40-4645-9faa-15796d52c7bd)


- **Change %:** The daily percentage change is centered around 0 but includes significant outliers on both ends, highlighting volatility in the market.

<img src="https://github.com/user-attachments/assets/0224b040-c80c-4e7c-a68c-62c6a8c2194f" width="700"/>


#### 3. Relationship Between Price and Volume

We explored how trading volume relates to Bitcoin price using a scatter plot. The plot reveals that while most trading occurs at lower volumes, some mid-to-high price levels correspond to higher trading activity.


  <img src="https://github.com/user-attachments/assets/69d04b88-02b1-41ca-ac97-4c312d13affa" alt="Scatter Plot - Price vs Volume" width="700"/>
</p>




#### 4. Stationarity and Trend Inspection
We used time series plots (not shown here) to visually assess stationarity and trends. This step informed the need for differencing in ARIMA modeling.

> These visualizations gave us a deeper understanding of the dataset's dynamics, helped detect anomalies, and guided preprocessing decisions.

 Time series modeling using ARIMA

### Train-Test Split

In time series forecasting, it is essential to preserve the chronological order of the data during model evaluation. Instead of a random split, we divide the data such that the model learns only from the past and is tested on the future — effectively simulating real-world forecasting scenarios.

In this project, we split the dataset chronologically:

Training Set: Covers 80% of the earlier portion of the dataset (January 13, 2016 – December 31, 2022)

Testing Set: Covers the remaining 20% (January 13, 2023 – April 19, 2025)

This method ensures that the future (test data) is never exposed during training, allowing us to properly evaluate how well the model generalizes and predicts unseen data.

![image](https://github.com/user-attachments/assets/c17efc51-a373-4cbb-8f4e-9322c20b7661)
>The visual representation above highlights the training and testing periods, helping us better understand the model's performance on future Bitcoin prices.




### Model Fitting and Outcome

Initially, we used the ARIMA (AutoRegressive Integrated Moving Average) model to capture temporal
dependencies. To enhance model performance, we introduced lag features (lag1 and lag2). Additionally, we
applied the AutoARIMA model to automatically select the best hyperparameters and improve prediction
accuracy.

#### ARIMA Model with Basic Configuration:
The ARIMA model is a statistical method used for time series forecasting. It consists of three main
components:

AR (AutoRegressive): Models the dependency between an observation and its lagged values.

I (Integrated): Makes the series stationary by removing trends and seasonality.

MA (Moving Average): Models the relationship between an observation and the residual error from the
moving average model.

Initially, we applied the basic ARIMA model to the Bitcoin price data. However, the performance could be
improved by capturing more temporal dependencies.

#### Output with Basic ARIMA Model:
![image](https://github.com/user-attachments/assets/4e81ee58-3183-40a5-b839-939a951ca6c0)
>The basic ARIMA model was able to capture the general trend in Bitcoin price fluctuations but lacked
accuracy in predicting short-term price changes.
&gt;RMSE: 55045.37
&gt;MAE: 50475.75

#### Introducing Lag Features for ARIMA:
To improve the ARIMA model, we introduced lag features (lag1 and lag2). These features represent the
Bitcoin prices from previous days:

Lag1: The Bitcoin price from the previous day.

Lag2: The Bitcoin price from two days ago.

Lag features help capture price momentum and dependencies between the current and past observations.
Including these features allows the model to better understand the time-series patterns and improve its
predictive accuracy.

#### Output with Lag Features for ARIMA:

By including lag1 and lag2 features, the ARIMA model became more sensitive to recent changes in Bitcoin
prices, improving the model’s predictive accuracy.

AutoARIMA for Hyperparameter Optimization:
After adding lag features, we further enhanced the model by running AutoARIMA. AutoARIMA is an
automated method that selects the best ARIMA model by optimizing the hyperparameters (p, d, q). It
analyzes the data and automatically identifies the optimal configuration for the model, ensuring better
performance and more accurate forecasts.

Prediction on Testing Data with AutoARIMA:

![image](https://github.com/user-attachments/assets/03e61dc8-c640-437c-a3d8-2470613f85e8)
>After training with AutoARIMA, the model performed significantly better on the testing dataset. The
predictions were more accurate, reflecting a deeper understanding of Bitcoin’s price patterns.

### Improved Future Price Prediction:
By incorporating lag features and running AutoARIMA, the model’s ability to predict future Bitcoin prices
improved considerably. The combination of temporal features and hyperparameter optimization allowed
the model to capture the long-term and short-term trends more effectively, producing better future price
predictions.

#### Prediction of Future Price with Lag Features and AutoARIMA:

![image](https://github.com/user-attachments/assets/3f1b70ed-216d-4379-b456-abe70bba98ca)
>The forecasted future Bitcoin prices showed an improvement in accuracy compared to earlier predictions,
especially in capturing sudden price fluctuations.

### LSTM Model:
We also explored the Long Short-Term Memory (LSTM) model as an alternative to ARIMA. LSTM is a type of
Recurrent Neural Network (RNN) that excels in learning long-term dependencies in time series data.

####  Challenges with LSTM:

Data Requirements: LSTM models require a large amount of training data. For better predictions, it&#39;s
essential to incorporate additional correlated variables, such as Ethereum price and the Dow Jones index,
alongside Bitcoin prices. However, in our case, the available data was limited, leading to underfitting of the
model.

Performance: Despite using the correlated variables, the LSTM model did not perform well, as the data was
insufficient for training the model effectively. As a result, the LSTM model did not deliver better predictions
compared to ARIMA.

# Conclusion:

ARIMA Model: The ARIMA model provided good performance in forecasting Bitcoin prices based on
historical data. It is well-suited for time series forecasting tasks with stationary data.

LSTM Model: Although LSTM is a powerful deep learning model for time series forecasting, the lack of
sufficient data and the need for additional correlated variables hindered its performance in this case.

##### Prediction of amount:
| **Date** | **Predicted Price** |
| ---------- | ------------------- |
| 2025-04-20 | 84,047.41 |
| 2025-04-21 | 84,508.75 |
| 2025-04-22 | 84,037.07 |
| 2025-04-23 | 81,054.96 |
| 2025-04-24 | 83,806.03 |
| 2025-04-25 | 84,068.92 |
| 2025-04-26 | 82,461.66 |
| 2025-04-27 | 83,769.38 |
| 2025-04-28 | 82,728.55 |
| 2025-04-29 | 86,521.28 |
| 2025-04-30 | 84,012.90 |
| 2025-05-01 | 83,873.82 |
| 2025-05-02 | 83,758.17 |
| 2025-05-03 | 85,950.54 |
| 2025-05-04 | 87,288.85 |
| 2025-05-05 | 87,183.91 |
| 2025-05-06 | 86,731.29 |
| 2025-05-07 | 86,898.78 |
| 2025-05-08 | 84,137.76 |
| 2025-05-09 | 82,447.85 |
| 2025-05-10 | 82,195.74 |
| 2025-05-11 | 82,420.09 |
| 2025-05-12 | 83,586.97 |

| 2025-05-13 | 84,352.28 |
| 2025-05-14 | 83,477.76 |
| 2025-05-15 | 83,886.65 |
| 2025-05-16 | 84,727.36 |
| 2025-05-17 | 84,313.11 |
| 2025-05-18 | 84,882.34 |
| 2025-05-19 | 85,097.76 |

### Model Evaluation
Evaluation Metrics:
The ARIMA model&#39;s performance was assessed using two key metrics:

Mean Absolute Error (MAE):
Value: 1065.28

Interpretation: The MAE represents the average absolute difference between the predicted and actual
Bitcoin prices. In this case, the model&#39;s predictions are, on average, off by approximately $1,065. Given
Bitcoin&#39;s volatility, this can be considered a reasonable error, but improvements can still be made to reduce
this discrepancy.

Root Mean Squared Error (RMSE):
Value: 1863.76

Interpretation: RMSE penalizes larger errors more heavily, making it a useful metric when high deviations in
price are critical to minimize. The model’s RMSE of ~1,864 suggests that, while the model is reasonably
accurate, there are larger fluctuations in the predicted prices that could be reduced. This is typical for
Bitcoin, as it experiences sharp price movements, but further model tuning could reduce these large errors.

Performance Summary:
Good Performance for Long-Term Forecasting: Considering Bitcoin’s highly volatile nature, an MAE of
~1,065 and RMSE of ~1,864 are acceptable for predicting future prices over long time horizons. However,
the accuracy could be improved for short-term predictions where precise forecasting is crucial.

Baseline Comparison: For a more meaningful evaluation, comparing this model’s performance against
simpler baseline methods (e.g., predicting the previous day&#39;s price) can help highlight how much the ARIMA
model improves on a straightforward approach.

### Future Work to Improve Model Performance
While the current model performs reasonably well, there are several opportunities for further improving
the predictions:

1. Incorporating More Features:
External Factors: Adding more features, such as Ethereum prices, trading volume, market sentiment, or
other financial indicators, could enhance the model’s predictive power. These variables are often
correlated with Bitcoin&#39;s price and can help capture additional patterns.

Lag Features: While lag1 and lag2 were introduced, experimenting with additional lags (e.g., lag3, lag4) or
other derived features (such as moving averages, price volatility, etc.) might improve the model&#39;s accuracy.

2. Expanding the Dataset:
The model could benefit from a larger dataset, particularly to capture more long-term trends or patterns.
Additional historical data, including global financial indicators or news events, could further improve
forecasting.

Data Augmentation techniques or synthetic data generation could also be considered if data availability is
an issue.

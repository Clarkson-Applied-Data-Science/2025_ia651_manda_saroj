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

### Data cleaning and preprocessing

Data cleaning and preprocessing are critical steps in any data science project, as they ensure the quality
and usability of the dataset. In this project, the following preprocessing tasks were performed:

Handling Non-Numeric Characters: The dataset contains columns like Price, Vol., and Change % that may
include non-numeric characters such as currency symbols ($), commas (,), and shorthand notation like &#39;K&#39;,
&#39;M&#39;, and &#39;B&#39; for thousands, millions, and billions, respectively. These need to be cleaned so that the data can
be used for further analysis and modeling.

Converting to Numeric Values: Once the non-numeric characters are removed, the columns are converted
to numeric data types. This is done to ensure that the values are interpretable by mathematical models like
ARIMA.

Handling Missing Values: After the conversion to numeric types, it&#39;s possible that some rows have missing
values (NaN) due to conversion issues or initial data gaps. These missing values are removed using the
dropna() function.

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

### Time series modeling using ARIMA

#### Train-Test Split

In time series forecasting, we split the data chronologically. We use the first part of the data as the training
set and the remaining part as the test set. This ensures that the future (test data) is never seen during the
model training process, mimicking real-world prediction scenarios.

The dataset is split into a training set (80%) and a testing set (20%) to evaluate the model&#39;s performance.
This is crucial in time series forecasting, as we need to train the model on past data and test its ability to
predict future values.

#### Fitting Model

The core objective of this project is to build a forecasting model to predict Bitcoin prices based on historical
data. For this task, the ARIMA (AutoRegressive Integrated Moving Average) model is used. ARIMA is a
widely used statistical method for time series forecasting, which is well-suited for predicting Bitcoin prices
due to its ability to capture temporal dependencies.

ARIMA Model: ARIMA uses three main components:Model order: (p=5, d=1, q=0)

AR (AutoRegressive): A regression model that uses the dependency between an observation and a number
of lagged observations.

I (Integrated): Differencing the series to make it stationary (i.e., to eliminate trends and seasonality).

MA (Moving Average): Models the dependency between an observation and a residual error from a moving
average model applied to lagged observations.

In time series forecasting, a train-test split is necessary to evaluate how well the model generalizes to
unseen data. Here’s the typical flow for ARIMA model evaluation:

### Forecasting future Bitcoin prices

Forecasting and Plotting: Finally, the predictions made by the model on the test set can be compared to the
actual values. We can visualize the results by plotting both the actual prices and the forecasted prices.

### Evaluating model performance using RMSE

Evaluation: After training the model on the training data, we evaluate the model on the test data using
performance metrics such as Mean Squared Error (MSE) or Mean Absolute Error (MAE). These metrics help
us assess how well the model performs on unseen data.
Mean Squared Error (MSE): It helps us understand how close the predicted values are to the actual values
in the test set. A lower MSE means better performance.

#### Actual Vs Prediction for bitcoin based on our model (auto ARIMA)
![image](https://github.com/user-attachments/assets/86a02ccd-7c81-4437-99fc-114ba490852d)

After fitting the model, we print the summary of the ARIMA model. This summary provides valuable
information such as:

The estimated parameters for the AR (AutoRegressive), MA (Moving Average), and I (Integrated)
components.

The statistical significance of the model parameters.

Performance metrics such as AIC (Akaike Information Criterion), which helps evaluate the goodness of the
model.

![image](https://github.com/user-attachments/assets/bcce2098-f5b0-4f54-93b8-001d7229730a)

### Residual Analysis and Model Diagnostics
After fitting the ARIMA model to predict Bitcoin prices, residual analysis is performed to evaluate the
model&#39;s performance. This analysis includes two key plots: the residual plot and the histogram of residuals.

![image](https://github.com/user-attachments/assets/2653b80b-d393-4028-b2af-817eee398f30)

#### Residual Plot
Purpose: Shows the residuals (errors) over time to check if the model captures all patterns in the data.

Interpretation: The residuals fluctuate around zero but show visible patterns (spikes around 2021),
indicating that the ARIMA model has not fully captured all trends and dependencies in the data.

Next Steps: Further model tuning or trying a Seasonal ARIMA (SARIMA) model to capture seasonal patterns
or exploring other models like LSTM could improve the forecasts.

#### Histogram of Residuals
Purpose: Displays the distribution of residuals to check if they follow a normal distribution (an assumption
of the model).

Interpretation: The residuals are skewed to the right, showing non-normality, which suggests that the
model may need further refinement.

Next Steps: Refining the model, possibly incorporating exogenous variables, or experimenting with LSTM
for non-linear trends might help improve the predictions.

Conclusion
The residual analysis indicates that the ARIMA model may not fully capture Bitcoin price movements.
Future work will focus on refining the model, experimenting with SARIMA, and exploring LSTM or hybrid
models for better forecasting.

# Store-sales-prediction
Python and machine learning project to predict sales from a store with historical sales and weather data and data-based features from an API.

## Table of contents

- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Tools](#tools)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Analysis](#data-analysis)
- [Results](#results)
- [Recommendations](#recommendations)
- [Limitations](#limitations)


## Project Overview 

This project aims to predict 15 days of sales for a convenience store using historical weather data and date-based features, using a weather API and Machine Learning model to identify trends and external factors, such as temperature, impact sales, and help determine optimal stock levels, purchases, human resources and opening times.


## Data Source

- The [sales data](https://github.com/diegoislasm/Store-sales-prediction/blob/main/store_sales.csv) was obtained through a real convenience store in Mexico City which was treated with multiplicative scaling beforehand to comply with data anonymization.
- The [historical weather](https://github.com/diegoislasm/Store-sales-prediction/blob/main/weather_mexico_city.csv) data and API was obtained from [Visual Crossing Weather](https://www.visualcrossing.com/). 


## Tools

- Python - Data preparation analysis
- XGBoost - Model

## Initial EDA

1. Check for null values.
2. Check data types.
3. Get summary statistics from datasets.

## Data Cleaning

1. Keep important columns, rename and format columns. 
2. Formate data column and set it as index.
3. Merge both dataframes on index.

## Data Preparation (feature engineering)

1. Label encoding
2. Create user defined functions for:
   - Basic date-based features.
   - Lag features.
   - Rolling average features.
3. Add all features with functions.


## Advanced EDA

-Plot the following to understand the data bettter:
   - Daily, weekly and monthly sales.
   - Correlation between sales and weather features.
   - Sales vs. weather in a scatter plot.
   - Sales vs weather over time.

```sql

```
## API

1. Set API variables to obtain weather data.
2. Create functions to get this weather data from the API:
   - 15 days of weather forecast.
   - Next 15 days of weather.
   - Update weather data for new sales data.

## Add new sales

- Function to let the user update new daily sales data.

## Model selection and training

- Train an XGBoost Regressor using TimeSeriesSplit (48 splits, test size = 15, gap = 7).

## Model evaluation

- Use the average of this metrics to evaluate model across folds:
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - Mean Absolute Percentage Error (MAPE)
   - R² Score

## Results

The model performance obtained the following results when predicting on test folds:

- Average RMSE: 3867.3485
- Average MAE: 3004.7112
- Average MAPE: 9.52%
- Average r2 Score: 0.2629

When predicting the next 15 days after the last day of the dataset the results were the following:


## Recommendations

Based on the analysis, I recommend the following actions:

## Limitations

- The oldest sales data available was from October 2020. 

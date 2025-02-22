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

- Python - Data preparation, cleaning and analysis.
- XGBoost - Machine Learning model to make predictions.

## Initial EDA

- Check for null values.
- Check data types.
- Get summary statistics from datasets.

## Data Cleaning

- Keep important columns, rename and format columns.
- Formate data column and set it as index.
- Merge both dataframes on index.

## Data Preparation (feature engineering)

- Label encoding
- Create user defined functions for:
   - Basic date-based features.
   - Holidays, Sunday and paydays features.
   - Lag features.
   - Rolling average features.
- Add all features with functions.


## Advanced EDA

- Plot the following to understand the data bettter:
   - Daily, weekly and monthly sales.
   - Correlation between sales and weather features.
   - Sales vs. weather in a scatter plot.
   - Sales vs weather over time.


## API

- Set API variables to obtain weather data.
- Create functions to get the following weather data from the API:
   - 15 days of weather forecast.
   - Next 15 days of weather.
   - Update weather data for new sales data.

## Add new sales

- Function to let the user update new daily sales data.

## Model selection and training

- Train an XGBoost Regressor using TimeSeriesSplit.

Here is an example of the code used:

```python
# Split the data for Time Series data
tss = TimeSeriesSplit(n_splits=48, test_size=15, gap = 7)
data = data.sort_index()

fold = 0
preds = []
rmse_scores = []
mae_scores = []
mape_scores = []
r2_scores = []

for train_idx, val_idx in tss.split(data):
  train = data.iloc[train_idx]
  test = data.iloc[val_idx]

  train = add_features(train)
  test = add_features(test)

  # Define features and target data
  FEATURES = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'precip',
       'precipcover', 'is_rain', 'windgust', 'windspeed', 'weekday', 'week',
       'day', 'month', 'year', 'day_of_year', 'is_holiday', 'is_sunday',
       'is_payday', 'is_holiday_1_lag', 'is_holiday_2_lag', 'is_holiday_3_lag',
       'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
       'sales_lag_364', 'sales_rolling_avg_7', 'sales_rolling_avg_14',
       'sales_rolling_avg_28', 'sales_rolling_avg_364']
  TARGET = 'sales'

  # Create x and y train and test dataset with features and target
  x_train = train[FEATURES]
  y_train = train[TARGET]

  x_test = test[FEATURES]
  y_test = test[TARGET]


  # Tune regressor model
  reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                         n_estimators=500,
                         early_stopping_rounds=50,
                         objective='reg:linear',
                         max_depth=3,
                         learning_rate=0.01)

  # Fit model on train and test sets
  reg.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=100)

  # Predict on test
  y_pred = reg.predict(x_test)
  preds.append(y_pred)

  # Calculate metrics
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  mae = mean_absolute_error(y_test, y_pred)
  mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
  r2 = r2_score(y_test, y_pred)

  # Store metrics for this fold
  rmse_scores.append(rmse)
  mae_scores.append(mae)
  mape_scores.append(mape)
  r2_scores.append(r2)
```

## Model evaluation

Use the average of these metrics to evaluate model across folds:
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

<img src="https://github.com/user-attachments/assets/9e07c4f0-2c19-410e-833d-0013087b83cb" height="400">


## Recommendations

Based on the analysis, I recommend the following actions:

## Limitations

- The oldest sales data available was from October 2020, having only 4 years of data for training.

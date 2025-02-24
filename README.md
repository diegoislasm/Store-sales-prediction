# Store-sales-prediction
Python and machine learning project to predict sales from a store with historical sales and weather data and data-based features from an API.

## Table of contents

- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Tools](#tools)
- [Initial EDA](#initial-eda)
- [Data Cleaning](#data-cleaning)
- [Data Preparation](#data-preparation)
- [Advanced EDA](#advanced-eda)
- [API](#api)
- [Model Selection and Training](#model-selection-and-training)
- [Results](#results)
- [Recommendations](#recommendations)
- [Limitations](#limitations)


## Project Overview 

This project aims to predict 15 days of sales for a convenience store using historical weather data and date-based features, using a weather API and Machine Learning model to identify trends and external factors, such as temperature, impact sales, and help determine optimal stock levels, purchases, human resources and opening times.


## Data Source

- The [sales data](https://github.com/diegoislasm/Store-sales-prediction/blob/main/store_sales.csv) was obtained through a real convenience store in Mexico City which was treated with multiplicative scaling beforehand to comply with data anonymization.
- The [historical weather](https://github.com/diegoislasm/Store-sales-prediction/blob/main/weather_mexico_city.csv) data and API data was obtained from [Visual Crossing Weather](https://www.visualcrossing.com/). 


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

## Data Preparation

Feature engineering using:
- Label encoding
- Create user defined functions for:
   - Basic date-based features.
   - Holidays, Sunday and paydays features.
   - Lag features.
   - Rolling average features.
- Add all features with functions.

This is an example of the code used to add the payday feature:
```python
def add_payday(df):
  '''
  Add is_payday feature to dataframe getting the 15th and last of the month which are paydays,
  adjusting to the previous Friday if payday is on a weekend and check if index matches a payday
  '''
  # Get the 15th and last day of the month
  fifteenth = df.index.to_period('M').to_timestamp('ms') + pd.offsets.Day(14)
  last_day = df.index.to_period('M').to_timestamp('M')

  # Adjust to the previous Friday if payday is on a weekend
  fifteenth = fifteenth.where(fifteenth.weekday <= 4, fifteenth - pd.offsets.Week(weekday=4))
  last_day = last_day.where(last_day.weekday <= 4, last_day - pd.offsets.Week(weekday=4))

  # Check if the index matches a payday
  df['is_payday'] = ((df.index.isin(fifteenth)) | (df.index.isin(last_day))).astype(int)

  return df
```

## Advanced EDA

- Plot the following to understand the data bettter:
   - Daily, weekly and monthly sales.
   - Correlation between sales and weather features.
    <img src="https://github.com/user-attachments/assets/d0c05ae9-7a6a-4b85-bf0f-1e95b082e400" height="400">

   - Sales vs. weather in a scatter plot.
   - Sales vs weather over time.


## API

- Set API variables to obtain weather data.
- Create functions to get the following weather data from the API:
   - 15 days of weather forecast.
   - Next 15 days of weather.
   - Update weather data for new sales data.

This in an example of the code used to get weather data from the API of the new sales data:
```python
def get_new_weather(df):
  '''
  Get weather data for new sales dates from API
  '''
  # Find the first and last date with weather missing values
  missing_tempmax = df[df['tempmax'].isna()]
  first_date = missing_tempmax.index.min().strftime('%Y-%m-%d')
  last_date = missing_tempmax.index.max().strftime('%Y-%m-%d')

  # Generate url and call API
  url = f"{BASE_URL}{CITY}/{first_date}/{last_date}?unitGroup=metric&include=days&key={API_KEY}&contentType=json"
  response = requests.get(url)
  API_data = response.json()

  # Format new weather data
  new_weather = pd.json_normalize(API_data['days'])
  new_weather['preciptype'] = new_weather['preciptype'].str[0]
  new_weather = clean_weather_data(new_weather)
  new_weather['date'] = pd.to_datetime(new_weather['date'])
  new_weather = new_weather.set_index('date')
  new_weather = new_weather.loc[missing_tempmax.index]

  # Merge new weather data with existing data
  original_columns = df.columns
  df = df.combine_first(new_weather)
  df = df.reindex(columns=original_columns)
  #df = add_features(df)

  print('Updated rows:')
  print(df.loc[missing_tempmax.index])
  return df
```

## Add new sales

- Function to let the user update new daily sales data and add the features to the new data.

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

Calculate the average of these metrics to evaluate the model across folds:
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - Mean Absolute Percentage Error (MAPE)
   - r2 Score

## Results

The model performance obtained the following results when predicting on test folds:

- Average RMSE: 3867.3485
- Average MAE: 3004.7112
- Average MAPE: 9.52%
- Average r2 Score: 0.2629

Based on this the model during training was off by by between 3,800 and 3,000 units of sale, or on average 9.52%, being very good numbers. While the r2 value shows that the model explains only 26% of the variance in sales, which is quite low.

When predicting the next 15 days after the last day of the dataset and comparing it to the real sales the results were the following:

<img src="https://github.com/user-attachments/assets/f2ab4967-ecc2-4cc8-ac09-69bdcd3f351a" height="400">


- RMSE: 10519.1949
- MAE: 8720.7644
- MAPE: 34.21%
- r2 Score: -7.7823

The model was off by between 10,500 and 8,700 when predicting the real future, or on average 34.21%, which is not very good. In this case the r2 score was negative which shows that the model is not capturing the actual trend.

## Recommendations

**Model recommendations:**

The model performance had good results on average during training but performed poorly when using all data and predicting the future. To improve this problem these changes could be implemented:
- Tune hyperparameters like complexity of the model and learning rate.
- Add more weather features or economic indicators.
- Add more lag features or remove non-important features.
- Use a different model like linear regression or ARIMA.


**Business recomendations:**

(For this recomendations the quality of the predictions is not going to be considered, only the results). 
Based on the predictions it is recommended to:
- Consider getting less staff or closing early from the 5th to the 8th given the low expected sales.
- Try clearance sales on the same period to compensate for the low sales on those days.
- Prepare a big stock for the days after the 11th because it is predicted to have 8 days in a row of high sales.
- Get more staff than usual for the same period to account for the extra customers.


## Limitations

- The oldest sales data available was from October 2020, having only 4 years of data for training.

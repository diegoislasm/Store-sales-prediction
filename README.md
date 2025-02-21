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


## Data Cleaning and Preparation

In this phase the tasks performed were the following:

1. Review datatypes and fix them.
2. Use UNION function to merge both datasets into temp table.
3. Check for null values.
4. Get the unique values for main fields.
5. Fix outliers.
6. Create temp tables to split data.
7. Create a CTE with a distinct value to use it as primary key.
8. Join temp tables using CTE to have a long format dataset.


## Exploratory Data Analysis

As part of the EDA I started by finding facts like the overall trending sales, the country with more EV sales, the country with more EVs, the most popular type of EV and more. To do this a combination of functions like GROUP BY, PARTITION BY, conditions, LAG and subqueries were used.

Here are some examples of the code used: 

```sql

```


## Data Analysis


## Results

The analysis results are summarized as follows:

## Recommendations

Based on the analysis, I recommend the following actions:

## Limitations

- The oldest sales data available was from October 2020. 

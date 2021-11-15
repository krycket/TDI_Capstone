import os
import requests
import pandas as pd
import numpy as np
import csv
from datetime import datetime, timedelta
from datetime import date
import plotly.express as px
import streamlit as st
import dill
from dill import dumps, loads
import matplotlib.pyplot as plt

import random
#from ipywidgets import widgets
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.compose import ColumnTransformer

max_data_date = datetime(2021, 10, 6) #last day for which there is shelter data available
ave_data_min = datetime(2021, 10, 7)
ave_data_max = datetime(2022, 10, 7)

#Get weather and wind chill data of interest from Visual Crossing
df_weather = pd.read_csv("VisualCrossingWeather.csv")
df_weather['date'] = pd.to_datetime(df_weather['Date time'])
df_weather = df_weather[df_weather['date'] < ave_data_min]
df_weather['Wind Chill'] = df_weather['Wind Chill'].fillna(df_weather['Minimum Temperature'])
df_weather['Precipitation'] = df_weather['Precipitation'].fillna(0.0)
df_weather['Snow'] = df_weather['Snow'].fillna(0.00)
df_weather['Heat Index'] = df_weather['Heat Index'].fillna(0.00)
df_weather['Snow Depth'] = df_weather['Snow Depth'].fillna(0.00)
df_weather['date'] = pd.to_datetime(df_weather['date'])
df_weather['day_of_year'] = df_weather.apply(lambda row: (row.date.day_of_year), axis=1)
df_weather['day'] = df_weather.apply(lambda row: int(row.date.day), axis=1)
df_weather['month'] = df_weather.apply(lambda row: int(row.date.month), axis=1)
df_weather['day_of_year_code'] = df_weather['month']*100 + df_weather['day']

#Process an average weather data set for the next year
df_ave_weather = df_weather.copy().groupby('day_of_year').mean()
new_dates = pd.date_range(start=ave_data_min,
                  end=ave_data_max,
                 freq='1d')
day_column = new_dates.day_of_year
day_column = new_dates.day_of_year
d = {'date':new_dates,'day_of_year':day_column}
df_future = pd.DataFrame(d)
df_future = df_future.merge(df_ave_weather, how='left', left_on='day_of_year', right_on='day_of_year')

df_wind = df_weather.copy()[['date', 'Wind Chill', 'Minimum Temperature', 'Maximum Temperature',
                           'Temperature', 'Precipitation', 'Snow']]
df_wind['Alert'] = 1.0
df_wind['Alert'][df_wind['Wind Chill'] > 32] = 0
df_wind = df_wind.dropna()
df_future['Alert'] = 1.0
df_future['Alert'][df_future['Wind Chill'] > 32] = 0
df_future['Total'] = 1200 #A holder value
df_future['Male'] = 100 #A holder value
df_future['Female'] = 100 #A holder value
df_future['dayWeek'] = df_future.apply(lambda row: int(row.date.day_of_week + 1), axis=1)
df_future['year'] = df_future.apply(lambda row: int(row.date.year), axis=1)
df_future['month'] = df_future.apply(lambda row: int(row.date.month), axis=1)
df_future['day'] = df_future.apply(lambda row: int(row.date.day), axis=1)
df_future = df_future.dropna()

#These are shelters we need to remove from Total, Male, and Female columns:
df_coh = pd.read_csv("InShelterPerDay_Gender_Shelter_10-06-2021.csv")
df_coh['date'] = pd.to_datetime(df_coh['date'])
mask = (df_coh['name'] == 'COH - The Triumph - DHS STFH - ES FAM - DHS Direct')
df_coh = df_coh[['date','Male', 'Female','name']].loc[mask]
df_coh = df_coh.rename(columns={"Male": "Male_COH", "Female": "Female_COH"})
df_coh['Total_COH'] = df_coh['Male_COH'] + df_coh['Female_COH']
df_coh = df_coh.drop(columns=['name'])

df_ken = pd.read_csv("InShelterPerDay_Gender_Shelter_10-06-2021.csv")
df_ken['date'] = pd.to_datetime(df_ken['date'])
mask = (df_ken['name'] == 'NCCF - The Kennedy - DHS STFH - ES FAM - DHS Direct')
df_ken = df_ken[['date','Male', 'Female','name']].loc[mask]
df_ken = df_ken.rename(columns={"Male": "Male_KEN", "Female": "Female_KEN"})
df_ken['Total_KEN'] = df_ken['Male_KEN'] + df_ken['Female_KEN']
df_ken = df_ken.drop(columns=['name'])

#Open shelter data again and remove two shelters above
df_use = pd.read_csv("InShelterPerDayGender-10-16-2021.csv")
df_use['date'] = pd.to_datetime(df_use['date'])
df_use['Total'] = df_use['Male'] + df_use['Female']

df_use = df_use.merge(df_coh, how='left', left_on='date', right_on='date')
df_use['Total'] = df_use['Total'] - df_use['Total_COH']
df_use['Male'] = df_use['Male'] - df_use['Male_COH']
df_use['Female'] = df_use['Female'] - df_use['Female_COH']
df_use = df_use.drop(columns=['Total_COH', 'Male_COH', 'Female_COH', 'NA', 'Transgender'])

df_use = df_use.merge(df_ken, how='left', left_on='date', right_on='date')
df_use['Total'] = df_use['Total'] - df_use['Total_KEN']
df_use['Male'] = df_use['Male'] - df_use['Male_KEN']
df_use['Female'] = df_use['Female'] - df_use['Female_KEN']
df_use = df_use.drop(columns=['Total_KEN', 'Male_KEN', 'Female_KEN'])


#Merge in temperature and windchill data from Visual Crossing
df_use = df_use.merge(df_wind, how='left', left_on='date', right_on='date')
df_use['date'] = pd.to_datetime(df_use['date'])
df_use = df_use[df_use['date'] > datetime(2007, 1, 1)]
df_use = df_use.drop(columns=['MinTempF', 'MaxTempF', 'SnowIn', 'PrecipIn'])

#Process average weather per day for future predictions (extend dates in df_use, then add future average weather data)
df_use = pd.concat([df_use, df_future])
df_use['day_of_year'] = df_use.apply(lambda row: (row.date.day_of_year), axis=1)
df_use['day_of_year_code'] = 100*df_use['month'] + df_use['day']
df_use = df_use[df_use['day_of_year_code'] != 229] #drop leap year day
df_use.loc[df_use.date >= max_data_date, "Total"] = 1101
df_use.loc[df_use.date >= max_data_date, "Male"] = 100
df_use.loc[df_use.date >= max_data_date, "Female"] = 100
df_use['day'] = df_use.apply(lambda row: int(row.date.day), axis=1)
df_use['month'] = df_use.apply(lambda row: int(row.date.month), axis=1)
df_use['year'] = df_use.apply(lambda row: int(row.date.year), axis=1)
df_use['dayWeek'] = df_use.apply(lambda row: int(row.date.day_of_week + 1), axis=1)
df_use['day_of_year'] = df_use.apply(lambda row: (row.date.day_of_year), axis=1)
df_use['day_of_year_code'] = 100*df_use['month'] + df_use['day']
df_use['day_of_year_code'][df_use['day_of_year_code'] == 229] = 228
df_use['Hypo'] = 0.0
df_use['Hypo'][df_use['month'] < 3] = 1.0
df_use['Hypo'][df_use['month'] > 11] = 1.0
df_use['FreezingAtEntry'] = 0.0
df_use['FreezingAtEntry'][df_use['Minimum Temperature'] > 32] = 1.0
df_use['prev_min_temp'] = df_use['Minimum Temperature'].shift(1)
df_use = df_use[df_use['date'] > datetime(2015, 1, 1)]

#Remove one outlier point:
df_use = df_use[df_use['Total'] >= 1100]

df_use = df_use.drop(['Snow Depth', 'Wind Speed', 'Wind Direction', 'Wind Gust', 'Visibility', 'Heat Index', 'Cloud Cover', 'Relative Humidity'], axis=1)

#Add Homeless population data and Metrobus usage:
df_use["dc_pop"] =  "582049"
df_use.loc[df_use.date >= datetime(2004, 6, 1), "dc_pop"] = "582049"
df_use.loc[df_use.date >= datetime(2005, 6, 1), "dc_pop"] = "585171"
df_use.loc[df_use.date >= datetime(2006, 6, 1), "dc_pop"] = "588292"
df_use.loc[df_use.date >= datetime(2007, 6, 1), "dc_pop"] = "588363"
df_use.loc[df_use.date >= datetime(2008, 6, 1), "dc_pop"] = "588433"
df_use.loc[df_use.date >= datetime(2009, 6, 1), "dc_pop"] = "605125"
df_use.loc[df_use.date >= datetime(2010, 6, 1), "dc_pop"] = "619624"
df_use.loc[df_use.date >= datetime(2011, 6, 1), "dc_pop"] = "633427"
df_use.loc[df_use.date >= datetime(2012, 6, 1), "dc_pop"] = "646669"
df_use.loc[df_use.date >= datetime(2013, 6, 1), "dc_pop"] = "658893"
df_use.loc[df_use.date >= datetime(2014, 6, 1), "dc_pop"] = "672228"
df_use.loc[df_use.date >= datetime(2015, 6, 1), "dc_pop"] = "681170"
df_use.loc[df_use.date >= datetime(2016, 6, 1), "dc_pop"] = "693972"
df_use.loc[df_use.date >= datetime(2017, 6, 1), "dc_pop"] = "702455"
df_use.loc[df_use.date >= datetime(2018, 6, 1), "dc_pop"] = "705749"
df_use.loc[df_use.date >= datetime(2019, 6, 1), "dc_pop"] = "689545"
df_use.loc[df_use.date >= datetime(2020, 6, 1), "dc_pop"] = "690345"
df_use["dc_pop"] = pd.to_numeric(df_use["dc_pop"])

#The following are from point in time surveys for individual homeless population, set in effect after hypothermia season ends
df_use["homeless_pop"] = "3200"
df_use.loc[df_use.date >= datetime(2011, 4, 21), "homeless_pop"] = 3228
df_use.loc[df_use.date >= datetime(2012, 4, 21), "homeless_pop"] = 3233
df_use.loc[df_use.date >= datetime(2013, 4, 21), "homeless_pop"] = 3128
df_use.loc[df_use.date >= datetime(2014, 4, 21), "homeless_pop"] = 3072
df_use.loc[df_use.date >= datetime(2015, 4, 21), "homeless_pop"] = 2889
df_use.loc[df_use.date >= datetime(2016, 4, 21), "homeless_pop"] = 2912
df_use.loc[df_use.date >= datetime(2017, 4, 21), "homeless_pop"] = 3156
df_use.loc[df_use.date >= datetime(2018, 4, 21), "homeless_pop"] = 3224
df_use.loc[df_use.date >= datetime(2019, 4, 21), "homeless_pop"] = 2971
df_use.loc[df_use.date >= datetime(2020, 4, 21), "homeless_pop"] = 2845
df_use.loc[df_use.date >= datetime(2021, 4, 21), "homeless_pop"] = 2936
df_use["homeless_pop"] = pd.to_numeric(df_use["homeless_pop"])


df_use["MetroBus"] = "0.46" #in millions of daily rides (mid-year ave.)
df_use.loc[df_use.date >= datetime(2000, 1, 1), "MetroBus"] = "0.46"
df_use.loc[df_use.date >= datetime(2015, 1, 1), "MetroBus"] = "0.44"
df_use.loc[df_use.date >= datetime(2016, 1, 1), "MetroBus"] = "0.41"
df_use.loc[df_use.date >= datetime(2017, 1, 1), "MetroBus"] = "0.37"
df_use.loc[df_use.date >= datetime(2018, 1, 1), "MetroBus"] = "0.36"
df_use.loc[df_use.date >= datetime(2019, 1, 1), "MetroBus"] = "0.35"
df_use.loc[df_use.date >= datetime(2020, 1, 1), "MetroBus"] = "0.35"
df_use.loc[df_use.date >= datetime(2020, 3, 1), "MetroBus"] = "0.18"
df_use.loc[df_use.date >= datetime(2020, 4, 1), "MetroBus"] = "0.0"
df_use.loc[df_use.date >= datetime(2020, 5, 1), "MetroBus"] = "0.02"
df_use.loc[df_use.date >= datetime(2020, 6, 1), "MetroBus"] = "0.02"
df_use.loc[df_use.date >= datetime(2020, 7, 1), "MetroBus"] = "0.03"
df_use.loc[df_use.date >= datetime(2020, 8, 1), "MetroBus"] = "0.03"
df_use.loc[df_use.date >= datetime(2020, 9, 1), "MetroBus"] = "0.04"
df_use.loc[df_use.date >= datetime(2020, 10, 1), "MetroBus"] = "0.04"
df_use.loc[df_use.date >= datetime(2020, 11, 1), "MetroBus"] = "0.04"
df_use.loc[df_use.date >= datetime(2020, 12, 1), "MetroBus"] = "0.04"
df_use.loc[df_use.date >= datetime(2021, 1, 1), "MetroBus"] = "0.11"
df_use.loc[df_use.date >= datetime(2021, 2, 1), "MetroBus"] = "0.10"
df_use.loc[df_use.date >= datetime(2021, 3, 1), "MetroBus"] = "0.12"
df_use.loc[df_use.date >= datetime(2021, 4, 1), "MetroBus"] = "0.12"
df_use.loc[df_use.date >= datetime(2021, 5, 1), "MetroBus"] = "0.13"
df_use.loc[df_use.date >= datetime(2021, 6, 1), "MetroBus"] = "0.14"
df_use.loc[df_use.date >= datetime(2021, 7, 1), "MetroBus"] = "0.14"
df_use.loc[df_use.date >= datetime(2021, 8, 1), "MetroBus"] = "0.15"
df_use.loc[df_use.date >= datetime(2021, 9, 1), "MetroBus"] = "0.12"
df_use.loc[df_use.date >= datetime(2021, 10, 1), "MetroBus"] = "0.23"
df_use["MetroBus"] = pd.to_numeric(df_use["MetroBus"])

#Add periodic features to be used for modeling
start = date(2005, 1, 1)
df_use['diff_date'] = df_use.apply(lambda row: (date(row.year, row.month, row.day) - start), axis=1)
df_use['Julian'] = pd.to_numeric(df_use['diff_date'].dt.days, downcast='integer')
df_use['sin(year)'] = np.sin(df_use['Julian'] / 365.25 * 2 * np.pi)
df_use['cos(year)'] = np.cos(df_use['Julian'] / 365.25 * 2 * np.pi)

#Save processed data

import dill
from dill import dumps, loads
dill.dump(df_use, open('df_use.pkd', 'wb'))


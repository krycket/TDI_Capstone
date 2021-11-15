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
import plotly.graph_objects as go
colors = px.colors.qualitative.Plotly
from plotly.subplots import make_subplots

import random
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

#st.set_page_config(layout="wide")

max_data_date = datetime(2021, 10, 6)
ave_data_min = datetime(2021, 10, 7)
ave_data_max = datetime(2022, 10, 7)

st.title("Predicting homeless shelter usage in a 'Right to Shelter' city.")
st.header("-K.L. Krycka")
st.subheader("This site is aimed at predicting walk-in (low-barrier) shelter usage for Washington DC.")
st.subheader("I. Shelter sites must be contracted well in advance of hypothermia season. Thus, is instructive to see how well the model performs for the remainder of the season after being trained on the latest shelter usage data at August 1st.")
st.write("To start, please select a year and gender (male, female, or total):")
year = st.selectbox(
     'Year:',
     ('2018', '2019', '2020', '2021'))

population = st.selectbox(
     'Population:',
     ('Total', 'Male', 'Female'))

winter_test_year = int(year)
Population_type = population

y = year

#st.write('You have currently selected:', population, ' for ', year)

df_use = dill.load(open('df_use.pkd', 'rb'))


def reduce_XY(time_start, time_finish, min_temp, max_temp, X_columns, Population_type, df_chosen):
    
    df_reduced = df_chosen.copy()
    leapyear_year = time_finish.year
    if leapyear_year%4 == 0:
        leap_date = datetime(leapyear_year, 2, 29)
        df_reduced = df_reduced[df_reduced['date'] != leap_date]
    df_reduced[Population_type] = df_reduced[Population_type] / df_reduced['homeless_pop']
    mask = (df_reduced['date'] >= time_start) & (df_reduced['date'] < time_finish) & (df_reduced['Minimum Temperature'] >= min_temp) & (df_reduced['Minimum Temperature'] < max_temp)
    df_reduced = df_reduced.copy().loc[mask]
    y = df_reduced.copy()[Population_type]
    df_reduced = df_reduced[X_columns]
    scale_by = df_reduced['homeless_pop'].tolist()
    
    return df_reduced['date'], y, scale_by
 

def reduce_dataframe(time_start, time_finish, min_temp, max_temp, X_columns, Population_type, df_chosen):
    
    df_reduced = df_chosen.copy()
    df_reduced[Population_type] = df_reduced[Population_type] / df_reduced['homeless_pop']
    mask = (df_reduced['date'] >= time_start) & (df_reduced['date'] < time_finish) & (df_reduced['Minimum Temperature'] >= min_temp) & (df_reduced['Minimum Temperature'] < max_temp)
    df_reduced = df_reduced.copy().loc[mask]
    y = df_reduced.copy()[Population_type]
    df_reduced = df_reduced[X_columns]
    scale_by = df_reduced['homeless_pop'].tolist()
    
    return df_reduced, y, scale_by

def compute_R2(y_pred1, y_pred2, y_true1, y_true2):
    
    y_train = []
    y_ave = 0
    for item in y_true2:
        y_train.append(item)
        y_ave += item
    for item in y_true1:
        y_train.append(item)
        y_ave += item
    y_ave = y_ave/(1.0*len(y_train))

    y_pred = []
    pred_ave =0
    for item in y_pred2:
        y_pred.append(item*1.0)
        pred_ave += item
    for item in y_pred1:
        y_pred.append(item*1.0)
        pred_ave += item
    pred_ave = pred_ave/(1.0*len(y_pred))

    diff_num = 0
    diff_denom= 0
    for i in range(0, len(y_train)):
        diff_num += (y_train[i] - y_pred[i])*(y_train[i] - y_pred[i])
        diff_denom += (y_train[i] - y_ave)*(y_train[i] - y_ave)

    R2= 1.0 - (diff_num / diff_denom)
    
    pred_diff = []
    std_diff = []
    for i in range(0,len(y_pred)):
        pred_diff.append(y_pred[i] - y_train[i])
        if y_pred[i] - y_train[i] < 200:
            std_diff.append(np.power(y_pred[i] - y_train[i],2))
        
    pred_bin = np.array(pred_diff)
    std = np.sqrt(sum(std_diff)/(1.0*len(y_pred)))

    return R2, std, y_ave, pred_ave, pred_bin

def plot_errors(showplot, name1, name2, color1, color2, y_pred_train1, y_pred_train2, y_true_train1, y_true_train2, y_pred_test1, y_pred_test2, y_true_test1, y_true_test2):

    R2_train, std_train, true_ave_train, pred_ave_train, train_bin = compute_R2(y_pred_train1, y_pred_train2, y_true_train1, y_true_train2)  
    R2_test, std_test, true_ave_test, pred_ave_test, test_bin = compute_R2(y_pred_test1, y_pred_test2, y_true_test1, y_true_test2)  
       
    if showplot > 0:

        st.write('The R-squared values for the', name1, ' and ', name2, ' data sets plotted above are ', round(R2_train,2), ' and ', round(R2_test,2), '. A better measure of the uncertainly on the number of daily shelter beds needed, however, is provided by the standard deviation (σ, calculated below).')

        title_hist = "Uncertainty evaluation for " + population + " population in " + year + "."

        df_train = pd.DataFrame(data = train_bin, columns = ['date'])
        df_test = pd.DataFrame(data = test_bin, columns = ['date'])
        x_label1 = name1 + " Model"
        x_label2 = name2 + " Model"
        num1 = str(round(std_train,2))
        num2 = str(round(std_test,2))

        extra1 = 'σ = ' + num1 + ' beds'
        extra2 = 'σ = ' + num2 + ' beds'

        fig = make_subplots(rows=1, cols=2, subplot_titles=(x_label1, x_label2), y_title='Number of Occurences', x_title = 'Predicted - Actual Daily Shelter Bed Usage')
        fig.add_trace(
            go.Histogram(x=df_train['date'], opacity=0.9, marker=dict(color=color1), name = extra1),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=df_test['date'], opacity=0.9, marker=dict(color=color2), name = extra2),
            row=1, col=2
        )
        fig.update_layout(height=350, width=750, title_text=title_hist) #title_text="Side By Side Subplots"
        st.write(fig)

    return

#@st.cache(suppress_st_warning=True)
def temp_modeling(time_start, time_finish, min_temp, max_temp, X_columns, categorical_columns, numerical_columns, Population_type, df_chosen):

    X_train, y_train, scale_by = reduce_dataframe(time_start, time_finish, min_temp, max_temp, X_columns, Population_type, df_chosen)

    preplowT = ColumnTransformer([
        ('categorical', OneHotEncoder(), categorical_columns),
        ('numeric', 'passthrough', numerical_columns),
        ('standard_scaler', StandardScaler(), numerical_columns),
        ('polynomial_trans', PolynomialFeatures(1), numerical_columns)
    ])
    model_byTemp = Pipeline([
        ('features', preplowT),
        ('regressor', RandomForestRegressor(max_depth=5, random_state=42))
    ])

    model = model_byTemp.fit(X_train,y_train)
    
    return model, scale_by

def SeasonPredictions(winter_test_year, Population_type):

    test_start_date = datetime(winter_test_year, 8, 1)
    test_end_date = datetime(winter_test_year+1, 4, 1)
    train_start_date = max(datetime(winter_test_year-4, 8, 1), datetime(2015, 8, 1))
    if Population_type == 'Female':
        train_start_date = max(datetime(winter_test_year-4, 8, 1), datetime(2016, 8, 1))
    train_end_date = test_start_date
    train_end_date = min(test_start_date, max_data_date)

    X_columns = ['date', 'dayWeek', 'homeless_pop', 'sin(year)', 'cos(year)', 'Wind Chill', 'Minimum Temperature', 'MetroBus', 'day_of_year', 'prev_min_temp', 'Snow', 'Precipitation']
    categorical_columns = ['dayWeek']
    numerical_columns = X_columns[2:]

    model_lowT, scaleby = temp_modeling(train_start_date, train_end_date, -150, 30, X_columns, categorical_columns, numerical_columns, Population_type, df_use) 
    model_midT, scaleby = temp_modeling(train_start_date, train_end_date, 30, 150, X_columns, categorical_columns, numerical_columns, Population_type, df_use) 

    if test_end_date < max_data_date:
        X_allT, y_true_allT, scale_allT = reduce_dataframe(train_start_date, test_end_date, -150,  150, X_columns, Population_type, df_use)
    else:
        X_allT, y_true_allT, scale_allT = reduce_dataframe(train_start_date, train_end_date, -150,  150, X_columns, Population_type, df_use)
        
    X_train_lowT, y_true_train_lowT, scale_by_train_lowT = reduce_dataframe(train_start_date, train_end_date, -150,  30, X_columns, Population_type, df_use)
    X_train_midT, y_true_train_midT, scale_by_train_midT = reduce_dataframe(train_start_date, train_end_date, 30,  150, X_columns, Population_type, df_use)

    X_test_lowT, y_true_test_lowT, scale_by_test_lowT = reduce_dataframe(test_start_date, test_end_date, -150,  30, X_columns, Population_type, df_use)
    X_test_midT, y_true_test_midT, scale_by_test_midT = reduce_dataframe(test_start_date, test_end_date, 30,  150, X_columns, Population_type, df_use)


    y_predict_train_lowT = model_lowT.predict(X_train_lowT)
    y_predict_train_midT = model_midT.predict(X_train_midT)

    y_predict_test_lowT = model_lowT.predict(X_test_lowT)
    y_predict_test_midT = model_midT.predict(X_test_midT)

    st.write('The data is modeled seperately for low temperature days (where the combination of minimum temperature plus windchill hits 30 F or lower) and mid temperature days (which never dip to 30 F or lower). The data is further segregated into training and test (prediction) data sets at the boundary of August 1 on the year selected above. Details of the model can be found at http://github.com/krycket/TDI_Capstone.')

    season_title1 = "Data and model for " + population + " population in " +  year + "."

    df_allT = pd.DataFrame(list(zip(X_allT['date'], y_true_allT*scale_allT)), columns =['date', 'Total'])
    df_train_lowT = pd.DataFrame(list(zip(X_train_lowT['date'], y_predict_train_lowT*scale_by_train_lowT)), columns =['date', 'Total'])
    df_train_midT = pd.DataFrame(list(zip(X_train_midT['date'], y_predict_train_midT*scale_by_train_midT)), columns =['date', 'Total'])
    df_test_lowT = pd.DataFrame(list(zip(X_test_lowT['date'], y_predict_test_lowT*scale_by_test_lowT)), columns =['date', 'Total'])
    df_test_midT = pd.DataFrame(list(zip(X_test_midT['date'], y_predict_test_midT*scale_by_test_midT)), columns =['date', 'Total'])
    fig = go.Figure()
    fig.add_traces(go.Scatter(x=df_allT['date'], y = df_allT['Total'], mode = 'lines+markers', marker = {'color' : 'blue'}, name = 'Actual Shelter Usage'))
    fig.add_traces(go.Scatter(x=df_train_lowT['date'], y = df_train_lowT['Total'], mode = 'markers', marker = {'color' : 'red', 'symbol' : 'square-open'}, name = 'Trained Model (low temp.)'))
    fig.add_traces(go.Scatter(x=df_train_midT['date'], y = df_train_midT['Total'], mode = 'markers', marker = {'color' : 'red'}, name = 'Trained Model (mid temp.)'))
    fig.add_traces(go.Scatter(x=df_test_lowT['date'], y = df_test_lowT['Total'], mode = 'markers', marker = {'color' : 'purple', 'symbol' : 'square-open'}, name = 'Prediction Model (low temp.)'))
    fig.add_traces(go.Scatter(x=df_test_midT['date'], y = df_test_midT['Total'], mode = 'markers', marker = {'color' : 'purple'}, name = 'Prediction Model (mid temp.)'))
    fig.update_xaxes(title_text = "Date", title_font = {"size": 20}, title_standoff = 25)
    fig.update_yaxes(tickangle = 0, title_text = "Daily Shelter Usage", title_font = {"size": 20}, title_standoff = 25)
    fig.update_layout(title=season_title1)
    st.write(fig)

    if test_end_date < max_data_date :
        name1 = 'Trained'
        name2 = 'Predicted'
        showplot = 1
        plot_errors(showplot, name1, name2, "red", "purple", y_predict_train_lowT*scale_by_train_lowT, y_predict_train_midT*scale_by_train_midT, y_true_train_lowT*scale_by_train_lowT, y_true_train_midT*scale_by_train_midT, 
                    y_predict_test_lowT*scale_by_test_lowT, y_predict_test_midT*scale_by_test_midT, y_true_test_lowT*scale_by_test_lowT, y_true_test_midT*scale_by_test_midT)


    test_start_winter = datetime(test_start_date.year, 11, 1)
    test_end_winter = datetime(test_end_date.year, 4, 1)
    prev_start_winter = datetime(test_start_date.year - 1, 11, 1)
    prev_end_winter = datetime(test_end_date.year - 1, 4, 1)

    X_prev_year, y_prev_year, scale_prev_year = reduce_XY(prev_start_winter, prev_end_winter, -150,  150, X_columns, Population_type, df_use)
    X_this_year, y_this_year, scale_this_year = reduce_XY(test_start_winter, test_end_winter, -150,  150, X_columns, Population_type, df_use)

    y_naive = y_prev_year*scale_this_year
    X_test_lowT, y_true_test_lowT, scale_by_test_lowT = reduce_dataframe(test_start_winter, test_end_winter, -150,  30, X_columns, Population_type, df_use)
    X_test_midT, y_true_test_midT, scale_by_test_midT = reduce_dataframe(test_start_winter, test_end_winter, 30,  150, X_columns, Population_type, df_use)
    y_predict_test_lowT = model_lowT.predict(X_test_lowT)
    y_predict_test_midT = model_midT.predict(X_test_midT)


    st.subheader("II. Correctly predicting the winter season is of primary importance. For evaluation, the trained model is compared to a naive model which scales the previous year's shelter bed usage by the current homeless population.")

    #st.write("Winter-centric data and model for ", population, " in ", year, "(selections listed near the top of the page):")

    season_title2 = "Winter-centric model for " + population + " population in " +  year + "."

    if test_end_winter < max_data_date:
        df_thisyear = pd.DataFrame(list(zip(X_this_year, y_this_year*scale_this_year)), columns =['date', 'Total'])
    elif test_end_winter > max_data_date:
        if test_start_winter < max_data_date:
            X_this_year, y_this_year, scale_this_year = reduce_XY(test_start_winter, max_data_date, -150,  150, X_columns, Population_type, df_use) 
            df_thisyear = pd.DataFrame(list(zip(X_this_year, y_this_year*scale_this_year)), columns =['date', 'Total'])
    df_test_lowT = pd.DataFrame(list(zip(X_test_lowT['date'], y_predict_test_lowT*scale_by_test_lowT)), columns =['date', 'Total'])
    df_test_midT = pd.DataFrame(list(zip(X_test_midT['date'], y_predict_test_midT*scale_by_test_midT)), columns =['date', 'Total'])
    df_naive = pd.DataFrame(list(zip(X_this_year, y_naive)), columns =['date', 'Total'])
    fig = go.Figure()
    if test_end_winter < max_data_date:
        fig.add_traces(go.Scatter(x=df_thisyear['date'], y = df_thisyear['Total'], mode = 'lines+markers', marker = {'color' : 'blue'}, name = 'Actual Shelter Population'))
    elif test_end_winter > max_data_date:
        if test_start_winter < max_data_date:
            fig.add_traces(go.Scatter(x=df_thisyear['date'], y = df_thisyear['Total'], mode = 'lines+markers', marker = {'color' : 'blue'}, name = 'Actual Shelter Population'))
    fig.add_traces(go.Scatter(x=df_test_lowT['date'], y = df_test_lowT['Total'], mode = 'markers', marker = {'color' : 'purple', 'symbol' : 'square-open'}, name = 'Prediction Model (low temp.)'))
    fig.add_traces(go.Scatter(x=df_test_midT['date'], y = df_test_midT['Total'], mode = 'markers', marker = {'color' : 'purple'}, name = 'Prediction Model (mid temp.)'))
    fig.add_traces(go.Scatter(x=df_naive['date'], y = df_naive['Total'], mode = 'markers', marker = {'color' : 'gray'}, name = 'Naive Prediction'))
    fig.update_xaxes(title_text = "Date", title_font = {"size": 20}, title_standoff = 25)
    fig.update_yaxes(tickangle = 0, title_text = "Shelter Usage", title_font = {"size": 20}, title_standoff = 25)
    fig.update_layout(title=season_title2)
    st.write(fig)

    if test_end_date < max_data_date :
        name2 = 'Naive'
        name1 = 'Predicted'
        showplot = 1
        plot_errors(showplot, name1, name2, "purple", "gray",
                    y_predict_test_lowT*scale_by_test_lowT, y_predict_test_midT*scale_by_test_midT, y_true_test_lowT*scale_by_test_lowT, y_true_test_midT*scale_by_test_midT,
                   y_naive, [], y_this_year*scale_this_year, [])

    st.write("As expected, the trained model generally outperforms the naive model.")
    
    return

def TenDayPredictions(start_year, start_month, start_day, Population_type):
    

    test_start_date = datetime(start_year, start_month, start_day)
    test_end_date = test_start_date + timedelta(days=10)
    
    prev_start_date = datetime(start_year - 1, start_month, start_day)
    prev_end_date = prev_start_date + timedelta(days=10)
    
    train_start_date = max(datetime(start_year-4, start_month, start_day), datetime(2015, 8, 1))
    if Population_type == 'Female':
        train_start_date = max(datetime(start_year-4, start_month, start_month), datetime(2016, 8, 1))
    train_end_date = test_start_date
    train_end_date = min(train_end_date, max_data_date)

    X_columns = ['date', 'dayWeek', 'homeless_pop', 'sin(year)', 'cos(year)', 'Wind Chill', 'Minimum Temperature', 'MetroBus', 'day_of_year', 'prev_min_temp', 'Snow', 'Precipitation']
    categorical_columns = ['dayWeek']
    numerical_columns = X_columns[2:]

    X_allT, y_true_allT, scale_allT = reduce_dataframe(test_start_date, test_end_date, -150,  150, X_columns, Population_type, df_use)

    X_train_lowT, y_true_train_lowT, scale_by_train_lowT = reduce_dataframe(train_start_date, train_end_date, -150,  30, X_columns, Population_type, df_use)
    X_train_midT, y_true_train_midT, scale_by_train_midT = reduce_dataframe(train_start_date, train_end_date, 30,  150, X_columns, Population_type, df_use)

    X_test_lowT, y_true_test_lowT, scale_by_test_lowT = reduce_dataframe(test_start_date, test_end_date, -150,  30, X_columns, Population_type, df_use)
    X_test_midT, y_true_test_midT, scale_by_test_midT = reduce_dataframe(test_start_date, test_end_date, 30,  150, X_columns, Population_type, df_use)

    model_lowT, scaleby = temp_modeling(train_start_date, train_end_date, -150, 30, X_columns, categorical_columns, numerical_columns, Population_type, df_use) 
    model_midT, scaleby = temp_modeling(train_start_date, train_end_date, 30, 150, X_columns, categorical_columns, numerical_columns, Population_type, df_use) 

    y_predict_train_lowT = model_lowT.predict(X_train_lowT)
    y_predict_train_midT = model_midT.predict(X_train_midT)
    
    X_prev_year, y_prev_year, scale_prev_year = reduce_XY(prev_start_date, prev_end_date, -150,  150, X_columns, Population_type, df_use)
    X_this_year, y_this_year, scale_this_year = reduce_XY(test_start_date, test_end_date, -150,  150, X_columns, Population_type, df_use)
    y_naive = y_prev_year*scale_this_year


    if len(X_test_lowT) > 0:
        y_predict_test_lowT = model_lowT.predict(X_test_lowT)
    else:
        y_predict_test_lowT = []
        scale_by_test_lowT = 1
        
    y_predict_test_midT = model_midT.predict(X_test_midT)
    
    
    fig = go.Figure()
    if test_end_date < max_data_date:
        df_allT = pd.DataFrame(list(zip(X_allT['date'], y_true_allT*scale_allT)), columns =['date', 'Total'])
        fig.add_traces(go.Scatter(x=df_allT['date'], y = df_allT['Total'], mode = 'lines+markers', marker = {'color' : 'blue'}, name = 'Actual Usage'))
    elif test_end_date > max_data_date:
        if test_start_date < max_data_date:
            X_allT, y_true_allT, scale_allT = reduce_dataframe(test_start_date, max_data_date, -150,  150, X_columns, Population_type, df_use)
            df_allT = pd.DataFrame(list(zip(X_allT['date'], y_true_allT*scale_allT)), columns =['date', 'Total'])
            fig.add_traces(go.Scatter(x=df_allT['date'], y = df_allT['Total'], mode = 'lines+markers', marker = {'color' : 'blue'}, name = 'Actual Usage'))

    st.write('This model utilizes shelter data that has been updated as of October 7, 2021. Low and/or Mid temperature points are shown if present in the selected 10 day viewing window. Short-range predictions utilizing a recently updated data set are likely to outperform corresponding naive models.')

    title_10day = "Ten day prediction for " + population_10day + " population starting on " + str(start_month) + " / " + str(start_day) + " / " + str(start_year) + "."

    df_naive = pd.DataFrame(list(zip(X_this_year, y_naive)), columns =['date', 'Total'])  
    fig.add_traces(go.Scatter(x=df_naive['date'], y = df_naive['Total'], mode = 'markers', marker = {'color' : 'gray'}, name = 'Naive prediction'))

    if len(X_test_lowT) > 0:
        df_test_lowT = pd.DataFrame(list(zip(X_test_lowT['date'], y_predict_test_lowT*scale_by_test_lowT)), columns =['date', 'Total'])
        fig.add_traces(go.Scatter(x=df_test_lowT['date'], y = df_test_lowT['Total'], mode = 'markers', marker = {'color' : 'purple'}, name = 'Predicted Usage (low temp.)'))
    df_test_midT = pd.DataFrame(list(zip(X_test_midT['date'], y_predict_test_midT*scale_by_test_midT)), columns =['date', 'Total'])
    fig.add_traces(go.Scatter(x=df_test_midT['date'], y = df_test_midT['Total'], mode = 'markers', marker = {'color' : 'purple'}, name = 'Predicted Usage (mid temp.)'))

    fig.update_xaxes(title_text = "Date", title_font = {"size": 20}, title_standoff = 25)
    fig.update_yaxes(tickangle = 0, title_text = "Shelter Usage", title_font = {"size": 20}, title_standoff = 25)
    fig.update_layout(title=title_10day)
    #fig.show()
    st.write(fig)

    name2 = 'Naive'
    name1 = "Precicted"
    showplot = 0
    if test_end_date < max_data_date:
        plot_errors(showplot, name1, name2, "c", "k",
                    y_predict_test_lowT*scale_by_test_lowT, y_predict_test_midT*scale_by_test_midT, y_true_test_lowT*scale_by_test_lowT, y_true_test_midT*scale_by_test_midT,
                     y_naive, [], y_this_year*scale_this_year, [])
        
    return


SeasonPredictions(winter_test_year, Population_type)


st.subheader('III. An additional use for the model is to refine the predicted shelter usage (and potentially invoke overflow shelter space) for the upcoming ten days.')

starting_date = datetime(2021, 9, 15)
cal_start = datetime(2018, 1, 1)
cal_end = ave_data_max - timedelta(11)
date_selected = st.date_input('Start of 10-day prediction', value=starting_date, min_value=cal_start, max_value=cal_end)
st.write(date_selected)

population_10day = st.selectbox(
     'Population_10DayPrediction:',
     ('Total', 'Male', 'Female'))

TenDayPredictions(date_selected.year, date_selected.month, date_selected.day, population_10day)

df_mini = df_use.copy()
df_mini = df_mini[df_mini['Male'] > 500]
ytotal_fulltime = df_mini['Total']
ymale_fulltime = df_mini['Male']
yfemale_fulltime = df_mini['Female']
X_fulltime = df_mini['date']

if st.checkbox('Click here to view full training sets.'): 

    df_data_total = pd.DataFrame(list(zip(X_fulltime, ytotal_fulltime)), columns =['date', 'Total'])
    df_data_male = pd.DataFrame(list(zip(X_fulltime, ymale_fulltime)), columns =['date', 'Total'])
    df_data_female = pd.DataFrame(list(zip(X_fulltime, yfemale_fulltime)), columns =['date', 'Total'])
    fig = go.Figure()
    fig.add_traces(go.Scatter(x=df_data_total['date'], y = df_data_total['Total'], mode = 'lines', line=dict(color="blue"), name = 'Total'))
    fig.add_traces(go.Scatter(x=df_data_total['date'], y = df_data_male['Total'], mode = 'lines', line=dict(color="red"), name = "Male"))
    fig.add_traces(go.Scatter(x=df_data_total['date'], y = df_data_female['Total'], mode = 'lines', line=dict(color="purple"), name = 'Female'))
    st.write(fig)

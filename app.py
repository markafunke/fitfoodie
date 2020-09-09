#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 12:21:12 2020

@author: markfunke
"""
import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import datetime


# MAIN PAGE HEADINGS
st.title('Fit Foodie Finds Forecaster')

# SIDE BAR
st.sidebar.markdown('#### Alter Forecast View:')

forecast_type = st.sidebar.radio(
"Forecast Type:",
("RPM", "Pageviews", "Earnings")
)

# forecast_type = st.sidebar.radio(
# "Forecast Type:",
# ("RPM", "Pageviews", "Earnings")
# )

# st.sidebar.markdown('#### Show Historical Comparisons:')
cb = st.sidebar.checkbox("Show Low & High Forecast",value=False,key="Test 1")
# b = st.sidebar.checkbox("Last Week",value=False,key="Test 1")
# c = st.sidebar.checkbox("This Week",value=False,key="Test 1")

# time_range = st.sidebar.radio(
# "Historical or Future:",
# ("Historical", "Future")
# )

time_period = st.sidebar.radio(
"Forecast Horizon:",
("Week", "Month", "Year")
)

st.sidebar.markdown('#### Alter Forecast Options:')

trend = st.sidebar.slider('Yearly Trend Adjustment', 0, 100, 5)

covid_shutdown = st.sidebar.radio(
"Model Future COVID Shutdown?",
("Yes", "No")
)

date = st.sidebar.date_input(
    'COVID Shutdown Start Date', datetime.date(2021,1,1)
    )

date = st.sidebar.date_input(
    'COVID Shutdown End Date', datetime.date(2021,2,1)
    )



# FUNCTIONS
@st.cache
def get_data():
    # rpm earnings data from AdThrive
    df_rpm = pd.read_csv('data/Earnings_2017-05-01_2020-08-31.csv')
    df_rpm.rename(columns={"Start Date": "ds", "RPM": "y"}, inplace=True)
    df_rpm.ds = pd.to_datetime(df_rpm.ds)
    
    # views data from Google Analytics
    df_views = pd.read_csv('data/Total_Views_20150101-20200831.csv')
    df_views.rename(columns={"Day Index": "ds", "Pageviews": "y"}, inplace=True)
    df_views.ds = pd.to_datetime(df_views.ds)
    
    return df_rpm, df_views

@st.cache
def fit_predict():
    model_views = Prophet(holidays=holidays,seasonality_mode="multiplicative")
    model_views.fit(df_views)
    
    model_rpm = Prophet(holidays=holidays,seasonality_mode="multiplicative")
    model_rpm.fit(df_rpm)
    
    future_views = model_views.make_future_dataframe(periods=365)
    forecast_views = model_views.predict(future_views)
    
    future_rpm = model_rpm.make_future_dataframe(periods=365)
    forecast_rpm = model_rpm.predict(future_rpm)
    
    return forecast_views, forecast_rpm

@st.cache
def merge_forecast():
    # merge views and rpm together for forecast
    views_merge = forecast_views[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    rpm_merge = forecast_rpm[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    views_merge.rename(columns={"yhat": "views", "yhat_lower": "views_l", "yhat_upper": "views_h"}, inplace=True)
    rpm_merge.rename(columns={"yhat": "rpm", "yhat_lower": "rpm_l", "yhat_upper": "rpm_h"}, inplace=True)
    
    df = rpm_merge.merge(views_merge, how="left", on=["ds"])    
    df["rev"] = df["views"]/1000 * df["rpm"]
    df["rev_l"] = df["views_l"]/1000 * df["rpm_l"]
    df["rev_h"] = df["views_h"]/1000 * df["rpm_h"]
    
    df = df.merge(df_views, how="left", on=["ds"])
    df = df.merge(df_rpm, how="left", on=["ds"])
    df = df.rename(columns={"y_x":"views_true", "y_y":"rpm_true"})
    df["rev_true"] = df["views_true"]/1000 * df["rpm_true"]
    
    # Add datetime characterics for calculations
    df["day_of_week"] = df.ds.dt.day_name()
    df["week"] = df.ds.dt.isocalendar().week
    df["month"] = df.ds.dt.month
    df["year"] =df.ds.dt.year
    
    return df

# COVID caused unusually high March views that can't be expected as a normal March seasonality
covid = pd.DataFrame({
  'holiday': 'covid',
  'ds': pd.date_range('2020-03-18','2020-06-05', freq='1D'),
  'lower_window': 0,
  'upper_window': 1,
})

holidays = covid

# RUN MODEL
# Load Data
try:
    df_rpm, df_views = get_data()
except urllib.error.URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )
    
# Fit and Predict
forecast_views, forecast_rpm = fit_predict()

# merge views and rpm together for earnings forecast
df = merge_forecast()


# Plot vs same week last year
def plot_v_last_yr(df):
    newest_row = df.dropna().sort_values("ds",ascending=False).iloc[1]
    current_day = newest_row["ds"]
    current_week = newest_row["week"]
    current_year = newest_row["year"]
    
    
    # could just edit this to be a function that takes in timedelta and spits out a plot!!!!
    # DO THIS
    next_week_day = current_day + datetime.timedelta(weeks=1)
    last_week_day = current_day - datetime.timedelta(weeks=1)
    last_two_week_day = current_day - datetime.timedelta(weeks=2)
    last_year_day_high = current_day - datetime.timedelta(weeks=52)
    last_year_day_low = current_day - datetime.timedelta(weeks=53)
    
    next_wk = df[(df["ds"] > current_day) & (df["ds"] <= next_week_day)].reset_index()
    last_wk = df[(df["ds"] > last_week_day) & (df["ds"] <= current_day)].reset_index()
    last_wk_two = df[(df["ds"] > last_two_week_day) & (df["ds"] <= last_week_day)].reset_index()
    last_yr_wk = df[(df["ds"] > last_year_day_low) & (df["ds"] <= last_year_day_high)].reset_index()

    plt.scatter(next_wk.index,next_wk.views)
    plt.scatter(last_wk.index,last_wk.views_true)
    plt.scatter(last_wk_two.index,last_wk_two.views_true)      
    plt.scatter(last_yr_wk.index,last_yr_wk.views_true)
    plt.plot(next_wk.index,next_wk.views)
    plt.plot(last_wk.index,last_wk.views_true)
    plt.plot(last_wk_two.index,last_wk_two.views_true)
    plt.plot(last_yr_wk.index,last_yr_wk.views_true)
    plt.xticks(ticks=range(7),labels=last_wk.day_of_week)
    plt.title("Page Views")
    plt.legend(["Next Week - Proj","This Week","Last Week","Same Week - 2019"])
    st.pyplot();

plot_v_last_yr(df)


# def plot_v_last_yr(df):
#     newest_row = df.dropna().sort_values("ds",ascending=False).iloc[1]
#     current_day = newest_row["ds"]
#     current_week = newest_row["week"]
#     current_year = newest_row["year"]
    
    
#     # could just edit this to be a function that takes in timedelta and spits out a plot!!!!
#     # DO THIS
#     next_week_day = current_day + datetime.timedelta(weeks=1)
#     last_week_day = current_day - datetime.timedelta(weeks=1)
#     last_two_week_day = current_day - datetime.timedelta(weeks=2)
#     last_year_day_high = current_day - datetime.timedelta(weeks=52)
#     last_year_day_low = current_day - datetime.timedelta(weeks=53)
    
#     next_wk = df[(df["ds"] > current_day) & (df["ds"] <= next_week_day)].reset_index()
#     last_wk = df[(df["ds"] > last_week_day) & (df["ds"] <= current_day)].reset_index()
#     last_wk_two = df[(df["ds"] > last_two_week_day) & (df["ds"] <= last_week_day)].reset_index()
#     last_yr_wk = df[(df["ds"] > last_year_day_low) & (df["ds"] <= last_year_day_high)].reset_index()

#     plt.scatter(next_wk.index,next_wk.views)
#     plt.scatter(last_wk.index,last_wk.views_true)
#     plt.scatter(last_wk_two.index,last_wk_two.views_true)      
#     plt.scatter(last_yr_wk.index,last_yr_wk.views_true)
#     plt.plot(next_wk.index,next_wk.views)
#     plt.plot(last_wk.index,last_wk.views_true)
#     plt.plot(last_wk_two.index,last_wk_two.views_true)
#     plt.plot(last_yr_wk.index,last_yr_wk.views_true)
#     plt.xticks(ticks=range(7),labels=last_wk.day_of_week)
#     plt.title("Page Views")
#     plt.legend(["Next Week - Proj","This Week","Last Week","Same Week - 2019"])
#     st.pyplot();

# plot_v_last_yr(df)






















# Plot results based on user input
# def plot(forecast_type):
    



    
    


 
    
 
    
 
#     plt.scatter(earnings_df.ds,earnings_df.rev_actual)
#     plt.scatter(earnings_df.ds,earnings_df.rev)
#     st.pyplot();
    
# plot(forecast_type)


# plt.plot(earnings_df.ds,earnings_df.rev_actual)
# plt.plot(earnings_df.ds,earnings_df.rev);



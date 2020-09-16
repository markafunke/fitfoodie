#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to run Fit Foodie Forecaster Streamlit app

@author: markafunke
"""
from utilities import (get_data, fit_predict, merge_forecast, df_between_dates
                    ,plotly_week, plotly_chart_wk, plotly_annual, plotly_chart_annual
                    ,top_post_dfs, top_post_compare, biggest_gainers)
import streamlit as st
from PIL import Image

# SIDE BAR BUTTONS
image = Image.open('fff.png') # fit foodie logo
st.sidebar.image(image, use_column_width=True)

st.sidebar.markdown("<h3 style='text-align: left; color: #00383E;'>Forecast View Options</h1>", unsafe_allow_html=True)

# select between 3 dashboard views
time_period = st.sidebar.radio(
"Dashboard View:",
("Weekly Comparison", "Post Comparison", "Annual Forecast")
)

# choose metric to display in "Weekly Comparison" and "Annual Forecast" views
metric = st.sidebar.radio(
"Metric:",
("Pageviews", "RPM", "Earnings")
)

# choose to display confidence interval in
# "Weekly Comparison" and "Annual Forecast" views
low_hi = st.sidebar.checkbox("Show Low & High Forecast",value=False)

# choose time period for comparison in "Post Comparison" view
comparison = st.sidebar.radio(
"Comparison Period (Post Comparison View Only)",
("Last Week", "Last Year")
)

# LOGIC TO DISPLAY CHARTS
# load latest data
df_rpm, df_views, df_holiday = get_data()
    
# fit prophet model and make 365 days of predictions
forecast_views, forecast_rpm = fit_predict(df_rpm,df_views,df_holiday)

# merge views and rpm together for earnings forecast
df = merge_forecast(forecast_rpm,forecast_views,df_rpm,df_views)

# create four dfs for weekly plot comparison
# next week, this week, last week, last year during the same week
next_wk = df_between_dates(df,0,1)
this_wk = df_between_dates(df,-1,0)
last_wk = df_between_dates(df,-2,-1)
last_yr_wk = df_between_dates(df,-53,-52)

# plot weekly comparison chart and percentage table
if time_period == "Weekly Comparison":
    plotly_week(metric,next_wk,this_wk,last_wk,last_yr_wk,low_hi)
    plotly_chart_wk(next_wk,this_wk,last_wk,last_yr_wk
                    ,metric,bkgrd_color = '#F9F8F5')

# create dfs for annual forecast plots
# past 26 weeks, and next 52 weeks
past = df_between_dates(df,-26,0) 
future = df_between_dates(df,-1,52)

# plot annual forecast chart, and annual forecast table summary
if time_period == "Annual Forecast":
    plotly_annual(metric,past,future,low_hi)
    plotly_chart_annual(df)

# create dfs containing daily post views for:
# yesterday, 1 week prior to yesterday, and 1 year prior to yesterday
y_df, lw_df, ly_df = top_post_dfs()

# plot top posts, biggest gainers, and biggest losers
if time_period == "Post Comparison":
    if comparison == "Last Week":
        top_post_compare(y_df,lw_df,"Last Week")
        biggest_gainers(y_df,lw_df,"Last Week", gain = True) #gainers
        biggest_gainers(y_df,lw_df,"Last Week", gain = False) #losers
    if comparison == "Last Year":
        top_post_compare(y_df,ly_df,"Last Year")
        biggest_gainers(y_df,ly_df,"Last Year", gain = True) #gainers
        biggest_gainers(y_df,ly_df,"Last Year", gain = False) #losers

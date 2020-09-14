#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 12:21:12 2020

@author: markfunke
"""

from utilities import (get_data, fit_predict, merge_forecast, df_between_dates
                    ,plotly_week, plotly_chart_wk, plotly_month, plotly_chart
                    ,fill, initialize_analyticsreporting, handle_report, get_report
                    ,top_posts, top_post_dfs, top_post_compare
                    ,biggest_gainers)
import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objs as go
from PIL import Image

# SIDE BAR
image = Image.open('fff.png')
st.sidebar.image(image, use_column_width=True)

st.sidebar.markdown("<h3 style='text-align: left; color: #00383E;'>Forecast View Options</h1>", unsafe_allow_html=True)

time_period = st.sidebar.radio(
"Dashboard View:",
("Weekly Comparison", "Post Comparison", "Annual Forecast")
)

metric = st.sidebar.radio(
"Metric:",
("Pageviews", "RPM", "Earnings")
)

low_hi = st.sidebar.checkbox("Show Low & High Forecast",value=False)

comparison = st.sidebar.radio(
"Comparison Period (Post Comparison View Only)",
("Last Week", "Last Year")
)

# Load latest data
df_rpm, df_views, df_holiday = get_data()
    
# Fit and Predict
forecast_views, forecast_rpm = fit_predict(df_rpm,df_views,df_holiday)

# merge views and rpm together for earnings forecast
df = merge_forecast(forecast_rpm,forecast_views,df_rpm,df_views)

# create dfs for weekly plot comparison   
next_wk = df_between_dates(df,0,1)
this_wk = df_between_dates(df,-1,0)
last_wk = df_between_dates(df,-2,-1)
last_yr_wk = df_between_dates(df,-53,-52)

# plot weekly comparison
if time_period == "Weekly Comparison":
    plotly_week(metric,next_wk,this_wk,last_wk,last_yr_wk,low_hi)
    plotly_chart_wk(next_wk,this_wk,last_wk,last_yr_wk
                    ,metric,bkgrd_color = '#F9F8F5')

# create dfs for annual forecast plots
past = df_between_dates(df,-26,0) 
future = df_between_dates(df,-1,52)

# plot monthly comparison
if time_period == "Annual Forecast":
    plotly_month(metric,past,future,low_hi)
    plotly_chart(df)

y_df, lw_df, ly_df = top_post_dfs()

if time_period == "Post Comparison":
    if comparison == "Last Week":
        top_post_compare(y_df,lw_df,"Last Week")
        biggest_gainers(y_df,lw_df,"Last Week", gain = True)
        biggest_gainers(y_df,lw_df,"Last Week", gain = False)
    if comparison == "Last Year":
        top_post_compare(y_df,ly_df,"Last Year")
        biggest_gainers(y_df,ly_df,"Last Year", gain = True)
        biggest_gainers(y_df,ly_df,"Last Year", gain = False)







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 12:21:12 2020

@author: markfunke
"""

from utilities import (get_data, fit_predict, merge_forecast, df_between_dates
                    ,plotly_week, plotly_chart_wk, plotly_month, plotly_chart
                    ,fill)
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
("Weekly Comparison", "Annual Forecast")
)

metric = st.sidebar.radio(
"Metric:",
("Pageviews", "RPM", "Earnings")
)

low_hi = st.sidebar.checkbox("Show Low & High Forecast",value=False)


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

 


import pandas as pd
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta

SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
KEY_FILE_LOCATION = './fff-time-series-d9e6714f8522.json'
VIEW_ID = '43760827'

def initialize_analyticsreporting():
  credentials = ServiceAccountCredentials.from_json_keyfile_name(
      KEY_FILE_LOCATION, SCOPES)
  analytics = build('analyticsreporting', 'v4', credentials=credentials)
  return analytics

#Get one report page
def get_report(analytics, pageTokenVar, start_date, end_date):
  return analytics.reports().batchGet(
      body={
        'reportRequests': [
        {
          'viewId': VIEW_ID,
          'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
          'metrics': [{'expression': 'ga:pageviews'}],
          'dimensions': [{'name': 'ga:pagePath'}],
          'pageSize': 10000,
          'pageToken': pageTokenVar,
          'samplingLevel': 'LARGE'
        }]
      }
  ).execute()
    
def handle_report(analytics,pagetoken,rows, start_date, end_date):  
    response = get_report(analytics, pagetoken, start_date, end_date)

    #Header, Dimentions Headers, Metric Headers 
    columnHeader = response.get("reports")[0].get('columnHeader', {})
    dimensionHeaders = columnHeader.get('dimensions', [])
    metricHeaders = columnHeader.get('metricHeader', {}).get('metricHeaderEntries', [])

    #Pagination
    pagetoken = response.get("reports")[0].get('nextPageToken', None)
    
    #Rows
    rowsNew = response.get("reports")[0].get('data', {}).get('rows', [])
    rows = rows + rowsNew
    print("len(rows): " + str(len(rows)))

    #Recursivly query next page
    if pagetoken != None:
        return handle_report(analytics,pagetoken,rows)
    else:
        #nicer results
        nicerows=[]
        for row in rows:
            dic={}
            dimensions = row.get('dimensions', [])
            dateRangeValues = row.get('metrics', [])

            for header, dimension in zip(dimensionHeaders, dimensions):
                dic[header] = dimension

            for i, values in enumerate(dateRangeValues):
                for metric, value in zip(metricHeaders, values.get('values')):
                    if ',' in value or ',' in value:
                        dic[metric.get('name')] = float(value)
                    else:
                        dic[metric.get('name')] = float(value)
            nicerows.append(dic)
        return nicerows

#Start
def top_posts(start_date,end_date):    
    analytics = initialize_analyticsreporting()
    
    global dfanalytics
    dfanalytics = []

    rows = []
    rows = handle_report(analytics,'0',rows,start_date,end_date)

    dfanalytics = pd.DataFrame(list(rows))
    # dfanalytics.sort_values(by=['ga:pageviews'], ascending=False).head(10)
    
    return dfanalytics

# if __name__ == '__main__':
#   main()
  
today = pd.to_datetime(datetime.date(datetime.now()))
yesterday = (today - timedelta(days = 1))
last_yr = (yesterday - timedelta(weeks = 52))
yesterday = yesterday.strftime('%Y-%m-%d')
last_yr = last_yr.strftime('%Y-%m-%d')

ly_df = top_posts(last_yr,last_yr)
y_df =   top_posts(yesterday,yesterday)

ly_df.rename(columns={"ga:pageviews": "views", "ga:pagePath": "post"}, inplace=True)
y_df.rename(columns={"ga:pageviews": "views", "ga:pagePath": "post"}, inplace=True)

# top posts df
top = pd.DataFrame()
top["Rank"] = range(1,11)
top["Yesterday"] = y_df.sort_values(by="views",ascending=False).reset_index().post
top["Views"] = y_df.sort_values(by="views",ascending=False).reset_index().views
top["Last Year"] = ly_df.sort_values(by="views",ascending=False).reset_index().post
top["Views "] = ly_df.sort_values(by="views",ascending=False).reset_index().views


# color_list = (
# [[bkgrd_color for val in range(3)],
# [bkgrd_color for val in range(3)],
# [bkgrd_color for val in range(3)],
# [bkgrd_color for val in range(3)]])

# create chart
fig5 = go.Figure(
    data=[go.Table(
    columnwidth = [12,80,18,80,18],
    header=dict(values=list(top.columns),
                fill_color='#00383E',
                font_color="white",
                align='left'),
    cells=dict(values=top.iloc[:,0:].T,
                fill_color= "#F9F8F5",
               # format = [None, ",.2f",None,None],
                # height = 20,
                align='left'))
    ])
fig5.layout.update(height=500, width = 1000, title = "Top Posts")

st.plotly_chart(fig5)



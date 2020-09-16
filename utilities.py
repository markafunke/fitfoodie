#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions used to pull views & earnings data, 
fit & predict a Prophet time series forecast, 
and manipulate and display a handful of graphs and tables in a Streamlit app

@author: markfunke
"""

import streamlit as st
import pandas as pd
from fbprophet import Prophet
import plotly.graph_objs as go
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta

@st.cache(allow_output_mutation=True)
def get_data():
    '''
    Returns latest earnings, views, and holiday data as dataframes

    '''
    # rpm earnings data from AdThrive
    df_rpm = pd.read_csv('data/Earnings.csv')
    df_rpm.rename(columns={"Start Date": "ds", "RPM": "y"}, inplace=True)
    df_rpm.ds = pd.to_datetime(df_rpm.ds)
    df_rpm["cap"] = 45
    df_rpm["floor"] = 0
    
    # views data from Google Analytics
    df_views = pd.read_csv('data/Total_Views.csv')
    df_views.rename(columns={"Day Index": "ds", "Pageviews": "y"}, inplace=True)
    df_views.ds = pd.to_datetime(df_views.ds)
    
    # Bring in holidays
    df_holiday = pd.read_csv('data/holidays.csv')
    
    return df_rpm, df_views, df_holiday

@st.cache(allow_output_mutation=True)
def fit_predict(df_rpm, df_views,df_holiday):
    """
    Fits Prophet model on views and RPM historical data, and returns
    dataframe with historical and 365 days of forecasted values
    
    Parameters
    ----------
    df_rpm : dataframe with rpm named "y", and datetime column named "ds"
    df_views : dataframe with views named "y", and datetime column named "ds"
    df_holiday : dataframe with holiday, ds, upper_window, lower_window columns
    See FB prophet documentation for details on requirements
    https://facebook.github.io/prophet/docs/quick_start.html
    
    """
    model_views = Prophet(
            growth = "linear",
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            holidays=df_holiday,
            seasonality_mode="multiplicative"
            ).add_seasonality(
                name="yearly",
                period = 365.25,
                fourier_order = 10
            ).add_seasonality(
                name = "weekly",
                period = 7,
                fourier_order = 7,
                prior_scale = 1
            )
    model_views.fit(df_views)
    
    model_rpm = Prophet(
            growth = "logistic",
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            holidays=df_holiday,
            seasonality_mode="multiplicative"
            # n_changepoints=10,
            ).add_seasonality(
                name="yearly",
                period = 365.25,
                fourier_order = 10,
            ).add_seasonality(
                name = "weekly",
                period = 7,
                fourier_order = 7
            ).add_seasonality(
                name = "quarterly",
                period = 365.25/4,
                fourier_order = 3
            )    
    model_rpm.fit(df_rpm)
    
    future_views = model_views.make_future_dataframe(periods=365)
    forecast_views = model_views.predict(future_views)
    
    future_rpm = model_rpm.make_future_dataframe(periods=365)
    future_rpm["cap"] = 55
    future_rpm["floor"] = 0
    forecast_rpm = model_rpm.predict(future_rpm)
    
    return forecast_views, forecast_rpm

@st.cache(allow_output_mutation=True)
def merge_forecast(forecast_rpm,forecast_views,df_rpm,df_views):
    """
    Takes outputs from get_data() and fit_predict() functions
    Merges together to create dataframes with forecasted and actual values
    for both views and RPM, and calculates revenue.

    Parameters
    ----------
    forecast_rpm : dataframe output from fit_predict()
    forecast_views : dataframe output from fit_predict()
    df_rpm : dataframe output from get_data()
    df_views : dataframe output from get_data()

    """
    
    # merge views and rpm together for forecast
    views_merge = forecast_views[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    rpm_merge = forecast_rpm[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    views_merge.rename(columns={"yhat": "views", "yhat_lower": "views_l", "yhat_upper": "views_h"}, inplace=True)
    rpm_merge.rename(columns={"yhat": "rpm", "yhat_lower": "rpm_l", "yhat_upper": "rpm_h"}, inplace=True)
    
    df = rpm_merge.merge(views_merge, how="left", on=["ds"])    
    
    # calculate forecasted revenue columns based on views and RPM
    df["rev"] = df["views"]/1000 * df["rpm"]
    df["rev_l"] = df["views_l"]/1000 * df["rpm_l"]
    df["rev_h"] = df["views_h"]/1000 * df["rpm_h"]
    
    # calculate actural revenue column based on views and RPM
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

def df_between_dates(df,wk_low,wk_high):
    """
    Outputs filtered dataframe within inputted week range
    , relative to most recent date

    Parameters
    ----------
    df : dataframe user wants to be filtered
    wk_low : integer, in weeks, low end of range
    wk_high : integer, in weeks, high end of range

    """
    # Find latest date with an observation to label as reference date
    newest_row = df.dropna().sort_values("ds",ascending=False).iloc[0] #latest date
    current_day = newest_row["ds"] 
    
    # Calculate low and high date compared to today
    low = current_day + timedelta(weeks=wk_low)
    high = current_day + timedelta(weeks=wk_high)
    
    # Create new dataframe limited to dates between low and high
    df_range = df[(df["ds"] > low) & (df["ds"] <= high)].reset_index()
    
    return df_range

def plotly_week(metric,next_wk,this_wk,last_wk,last_yr_wk,low_hi):
    """
    Plots overlaid daily chart comparing projection for next week to this week,
    last week, and the same week last year.

    Parameters
    ----------
    metric : string, "RPM", "Pageviews", or "Earnings"
            , which metric is displayed in plot
    next_wk : dataframe of next week views, rpm, revenue
    this_wk : dataframe of this week views, rpm, revenue
    last_wk : dataframe of last week views, rpm, revenue
    last_yr_wk : dataframe of last year views, rpm, revenue
    low_hi : Boolean, include low and high forecast lines for next week

    """
    # Functionality to alter chart based on user input in metric radio button
    # y_var is what is plotted
    # hov_f and prefix sets the format shown in the hover on the chart
    if metric == "RPM":
        y_var = "rpm"
        hov_f = '0.1f'
        prefix = '$'
    if metric == "Pageviews":
        y_var = "views"
        hov_f = '0,f'
        prefix = None
    if metric == "Earnings":
        y_var = "rev"
        hov_f = '0,f'
        prefix = '$'
    
    fig = go.Figure()
    
    # Create plot for each of 4 timeframes
    # Set color, dash type, and name of each plot
    plots = [next_wk,this_wk,last_wk,last_yr_wk]
    colors = ['#A4374E','#00383E','#F7D5C6','#BFD6D5']
    names = ["Next Week","This Week","Last Week","Last Year - Same Week"]
    dash = ["dot","solid","solid","solid"]
    for i in range(len(plots)):
        
        df = plots[i]
        color = colors[i]
        name = names[i]
        x = df.index
        y = eval(f"df.{y_var}")
        if i != 0:
            y = eval(f"df.{y_var}_true") # want actual values for every week but projection
        
        fig.add_trace(go.Scatter(
            x = x,
            y = y,
            mode = 'lines+markers',
            marker = dict(
                size=10,
                color=color,
                line=dict(width=1),
                opacity=0.6
            ),
            line = dict(
                width=5,
                color=color,
                dash=dash[i]
            ),
            name = name,
            text=df.ds.dt.strftime('%Y/%m/%d')))
        
    # Add two extra dotted lines plots for low and high projections if selected
    if low_hi:
        fig.add_trace(go.Scatter(
            x=next_wk.index, 
            y=eval(f"next_wk.{y_var}_l"), 
            name='Low/High Projection',
            line = dict(color='#A4374E', width=3, dash='dot')))
        
        fig.add_trace(go.Scatter(
            x=next_wk.index, 
            y=eval(f"next_wk.{y_var}_h"), 
            name = 'Low/High Projection',
            fill = "tonexty",
            line = dict(color='#A4374E', width=3, dash='dot')))

    # Set graph formatting
    fig.layout.update(dict(
        width = 800,
        title = f"Weekly {metric} Comparison",
        hovermode= 'x unified',
        plot_bgcolor='#F9F8F5',
        yaxis = dict(
            tickprefix = prefix,
            title = metric,
            hoverformat = hov_f),
        xaxis = dict(
            title = 'Day of Week',
            zeroline = False,
            tickmode = 'array',
            tickvals = [0,1,2,3,4,5,6],
            ticktext = last_yr_wk.day_of_week,
            tickformat= "%a")))
    
    # Post to streamlit
    st.plotly_chart(fig)
    
    
def plotly_chart_wk(next_wk,this_wk,last_wk,last_yr_wk,metric,bkgrd_color = '#F9F8F5'):
    """
    Plots percentage change table comparing projection for next week to this week,
    last week, and the same week last year.

    Parameters
    ----------
    metric : string, "RPM", "Pageviews", or "Earnings"
            , which metric is displayed in plot
    next_wk : dataframe of next week views, rpm, revenue
    this_wk : dataframe of this week views, rpm, revenue
    last_wk : dataframe of last week views, rpm, revenue
    last_yr_wk : dataframe of last year views, rpm, revenue
    bkgrnd_color: string, background color of table

    """    
    # set y variable based on metric input
    if metric == "RPM":
        y_var = "rpm"
    if metric == "Pageviews":
        y_var = "views"
    if metric == "Earnings":
        y_var = "rev"
    
    # Calculate % change vs this week for each day of week 
    df_wk = pd.DataFrame()
    df_wk["day_of_week"] = this_wk["day_of_week"]           
    df_wk["Week over Week"] = (this_wk[f"{y_var}_true"] / last_wk[f"{y_var}_true"] - 1) * 100
    df_wk["Year over Year"] = (this_wk[f"{y_var}_true"] / last_yr_wk[f"{y_var}_true"] - 1) * 100
    df_wk["Proj Next Week"] = (next_wk[f"{y_var}"] / this_wk[f"{y_var}_true"] - 1) * 100 
    
    # Format to percentage string for chart        
    df_wk['Week over Week']=df_wk['Week over Week'].map('{:,.1f}%'.format)
    df_wk["Year over Year"]=df_wk["Year over Year"].map('{:,.1f}%'.format)
    df_wk["Proj Next Week"]=df_wk["Proj Next Week"].map('{:,.1f}%'.format)
    
    # Shorten day_of_week column
    short_day = {"Sunday":"Sun"
                 ,"Monday":"Mon"
                 ,"Tuesday":"Tue"
                 ,"Wednesday":"Wed"
                 ,"Thursday":"Thu"
                 ,"Friday":"Fri"
                 ,"Saturday":"Sat"}
    df_wk["Day"] = df_wk["day_of_week"].apply(lambda x: short_day[x])
    
    # Format chart data
    df_wk = df_wk[["Day","Week over Week","Year over Year","Proj Next Week"]].T
    new_header = df_wk.iloc[0] #grab the first row for the header
    df_wk = df_wk[1:] #take the data less the header row
    df_wk.columns = new_header #set the header row as the df header
    df_wk = df_wk.reset_index()
    df_wk.rename(columns={"index":"Comparison Week"},inplace=True)    
    
    # set background color
    color_list = (
    [[bkgrd_color for val in range(3)],
    [bkgrd_color for val in range(3)],
    [bkgrd_color for val in range(3)],
    [bkgrd_color for val in range(3)]])
    
    # create chart
    fig_wk = go.Figure(
        data=[go.Table(
        columnwidth = [90,32,32,32,32,32,32,32],
        header=dict(values=list(df_wk.columns),
                    fill_color='#00383E',
                    font_color="white",
                    align='left'),
        cells=dict(values=df_wk.iloc[:,0:].T,
                    fill_color= color_list,
                   # format = [None, ",.2f",None,None],
                    height = 20,
                    align='left'))
        ])
    fig_wk.layout.update(height=275, width = 700,title = "Percentage Change")

    st.plotly_chart(fig_wk)
  
def fill(df,original,fill,new,keep_fill=False):
    """
    Helper function used to create a single column of both
    true past values and future forecasted values. Returns dataframe.

    Parameters
    ----------
    df : dataframe containing past and future columns
    original : column name of historic values used as base column
    fill : column name of future values used to "fill" future dates of original column
    new : name of new column containing both past and future values
    keep_fill : boolean, default False.
                Set to true to replace the "fill" column with the "new" column
    """
    df[new] = df[original]
    df[new].fillna(df[fill],inplace=True)
    if keep_fill:
        df[fill] = df[new]
    return df

  
def plotly_annual(metric,past,future,low_hi):
    """
    Plots time series of 7 day moving average historical and future values
    for selected metric

    Parameters
    ----------
    metric : string, "RPM", "Pageviews", or "Earnings"
            , which metric is displayed in plot
    past : dataframe of past views, rpm, revenue
    future : dataframe of future week views, rpm, revenue
    low_hi : Boolean, include low and high forecast lines

    """
    # Functionality to alter chart based on user input in metric radio button
    # y_var is what is plotted
    # hov_f and prefix sets the format shown in the hover on the chart
    if metric == "RPM":
        y_var = "rpm"
        hov_f = '.1f'
        prefix = '$'
    if metric == "Pageviews":
        y_var = "views"
        hov_f = '0,f'
        prefix = None
    if metric == "Earnings":
        y_var = "rev"
        hov_f = '0f'
        prefix = '$'
     
    # fills single overlapping week in "future" time period with past time
    # period values. This is so line chart connects at one point and 
    # appears to be continuous
    for item in ["views","rev","rpm"]:
        future = fill(future,f"{item}_true",f"{item}","temp",True)
        future = fill(future,f"{item}_true",f"{item}_l","temp",True)
        future = fill(future,f"{item}_true",f"{item}_h","temp",True)
        
    fig = go.Figure()
    
    # Create plot for past and future time periods
    # choose colors and name of each plot
    plots = [past,future]
    colors = ['#00383E','#A4374E']
    names = ['Historical','Forecast']
    for i in range(len(plots)):
        df = plots[i]
        color = colors[i]
        name = names[i]
        x = df.ds
        y = eval(f"df.{y_var}")
        if i == 0:
            y = eval(f"df.{y_var}_true") # want actual values for historical period
    
        fig.add_trace(go.Scatter(
            x = x,
            y = y.rolling(window=7).mean(), # plot 7 day rolling average
            mode = 'lines',
            marker = dict(
                size=10,
                color=color,
                line=dict(width=1),
                opacity=0.6
            ),
            name = name,
            text=df.ds.dt.strftime('%Y/%m/%d')))
        
    # Add two extra dotted lines plots for low and high projections if selected        
    if low_hi:
        fig.add_trace(go.Scatter(
            x=future.ds, 
            y=eval(f"future.{y_var}_l").rolling(window=7).mean(), 
            name='Low/High Projection',
            line = dict(color='#A4374E', width=1, dash='solid')))
        
        fig.add_trace(go.Scatter(
            x=future.ds, 
            y=eval(f"future.{y_var}_h").rolling(window=7).mean(), 
            name = 'Low/High Projection',
            fill = "tonexty",
            line = dict(color='#A4374E', width=1, dash='solid')))

    # Set graph formatting
    fig.layout.update(dict(
        title = f"7-Day Moving Average {metric} Forecast",
        hovermode = 'closest',
        plot_bgcolor='#F9F8F5',
        yaxis = dict(
            tickprefix = prefix,
            showgrid=False,
            title = metric,
            hoverformat = hov_f),
        xaxis = dict(
            showgrid=True,
            title = 'Date',
            zeroline = False
            # nticks=5,
            # tickformat='%b %y'
            )))

    # Post to streamlit
    st.plotly_chart(fig)


def plotly_chart_annual(df):
    """
    Plots a chart of the current year, summarized by month.
    Columns are RPM, Views, and Revenue for each month.
    Highlights future months in pink.

    Parameters
    ----------
    metric : dataframe output from merge_forecast()

    """    
    
    # calculate most recent day, month, and year
    newest_row = df.dropna().sort_values("ds",ascending=False).iloc[0]
    current_month = newest_row["ds"].month
    current_year = newest_row["ds"].year
    
    # Fill one column with all past and future values in order to sum by month
    # Currently past views are in one column and future are in another column
    # Currently there is a break from views to views_true once we move to the future
    for item in ["views","rev","rpm"]:
        df = fill(df,f"{item}_true",f"{item}",f"{item}_proj")
        df = fill(df,f"{item}_true",f"{item}_l",f"{item}_proj_l")
        df = fill(df,f"{item}_true",f"{item}_h",f"{item}_proj_h")
    
    # Group by month
    annual = (df.groupby(["year","month"])
                        ["views_proj","views_proj_l","views_proj_h",
                         "rev_proj","rev_proj_l","rev_proj_h"]).sum().reset_index()
    
    # Add a total column to the bottom of the chart
    total = (annual.groupby(["year"])["views_proj","views_proj_l","views_proj_h",
                         "rev_proj","rev_proj_l","rev_proj_h"]).sum().reset_index()
    annual = pd.concat([annual,total])
    
    # Back into RPM from revenue and views
    annual["rpm_proj"] = annual["rev_proj"] / (annual["views_proj"]/1000)
    annual["rpm_proj_h"] = annual["rev_proj_h"] / (annual["views_proj_h"]/1000)
    annual["rpm_proj_l"] = annual["rev_proj_l"] / (annual["views_proj_l"]/1000)
    
    # Format Yearmo, limit to just current year
    annual["Year_Month"] = pd.to_datetime(annual[["year","month"]].assign(DAY=1)).dt.strftime('%b %y')
    annual = annual[annual.year == current_year]
    annual.Year_Month.fillna("Total",inplace=True) #format total column 
    
    # Format numeric columns
    annual = annual[["Year_Month","views_proj","rpm_proj","rev_proj"]]
    annual['rev_proj']=annual['rev_proj'].map('${:,.0f}'.format)
    annual['rpm_proj']=annual['rpm_proj'].map('${:,.2f}'.format)
    annual['views_proj']=annual['views_proj'].map('{:,.0f}'.format)
    
    # Rename all columns for nice chart viewing
    annual.rename(columns={"Year_Month": "Month", "rev_proj": "Revenue"
                           , "views_proj":"Views", "rpm_proj": "RPM"},inplace=True)
    
    # Color row differently for all rows that have yet to be completed
    color_list = (
    [['#FBEBE3' if val >= current_month - 1 else '#F9F8F5' for val in range(11)],
    ['#FBEBE3' if val >= current_month - 1 else '#F9F8F5' for val in range(11)],
    ['#FBEBE3' if val >= current_month - 1 else '#F9F8F5' for val in range(11)],
    ['#FBEBE3' if val >= current_month - 1 else '#F9F8F5' for val in range(11)]])
    
    fig_month = go.Figure(
        data=[go.Table(
        header=dict(values=list(annual.columns),
                    fill_color='#00383E',
                    font_color="white",
                    align='left'),
        cells=dict(values=[annual.Month, annual.Views, annual.RPM, annual.Revenue],
                   fill_color= color_list,
                   # format = [None, ",.2f",None,None],
                   height = 20,
                   align='left'))
        ])
            
    fig_month.layout.update(height=475)

    st.plotly_chart(fig_month)

# Google API code all referenced from an example here:
# https://github.com/Zrce/Python-Pandas-SEO-Videos/blob/master/Analytics%20Big%20Beach%20Spring%20Break.ipynb

# Google API & view id reference by functions below
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
KEY_FILE_LOCATION = './fff-time-series-key.json'
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

def top_posts(start_date,end_date):    
    analytics = initialize_analyticsreporting()
    
    global dfanalytics
    dfanalytics = []

    rows = []
    rows = handle_report(analytics,'0',rows,start_date,end_date)

    dfanalytics = pd.DataFrame(list(rows))
    
    return dfanalytics
 
def top_post_dfs():
    """
    Leverages Google API to pull all daily posts and views for:
        yesterday, last week, and one year ago
    Returns a dataframe of each.

    """
    today = pd.to_datetime(datetime.date(datetime.now()))
    yesterday = (today - timedelta(days = 1))
    last_wk = (today - timedelta(weeks = 1))
    last_yr = (yesterday - timedelta(weeks = 52))
    yesterday = yesterday.strftime('%Y-%m-%d')
    last_wk = last_wk.strftime('%Y-%m-%d')
    last_yr = last_yr.strftime('%Y-%m-%d')
    
    ly_df = top_posts(last_yr,last_yr)
    lw_df = top_posts(last_wk,last_wk)
    y_df =  top_posts(yesterday,yesterday)
    
    y_df.rename(columns={"ga:pageviews": "views", "ga:pagePath": "post"}, inplace=True)
    ly_df.rename(columns={"ga:pageviews": "views", "ga:pagePath": "post"}, inplace=True)
    lw_df.rename(columns={"ga:pageviews": "views", "ga:pagePath": "post"}, inplace=True)
    
    return y_df, lw_df, ly_df

def top_post_compare(df_main,df_comp,comp_name):
    """
    Creates plotly table comparing top 10 posts and views from two time periods

    Parameters
    ----------
    df_main : dataframe output from top_post_dfs(), yesterday's views
    df_comp : dataframe output from top_post_dfs(), period to compare
    comp_name : string, name of comparison period

    """
    top = pd.DataFrame()
    top["Day Rank"] = range(1,11)
    top[f"{comp_name} Post"] = df_comp.sort_values(by="views",ascending=False).reset_index().post
    top[f"{comp_name} Views"] = df_comp.sort_values(by="views",ascending=False).reset_index().views    
    top["Yesterday Post"] = df_main.sort_values(by="views",ascending=False).reset_index().post
    top["Yesterday Views"] = df_main.sort_values(by="views",ascending=False).reset_index().views

    top[f"{comp_name} Views"]=top[f"{comp_name} Views"].map('{:,.0f}'.format)   
    top["Yesterday Views"]=top["Yesterday Views"].map('{:,.0f}'.format)

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
    
def biggest_gainers(df_main,df_comp,comp_name,gain = True):
    """
    Creates plotly table comparing top 10 % gain or loss of yesterday's posts
    to posts from a prior time period. Minimum 100 views in prior period
    to qualify.

    Parameters
    ----------
    df_main : dataframe output from top_post_dfs(), yesterday's views
    df_comp : dataframe output from top_post_dfs(), period to compare
    comp_name : string, name of comparison period
    gain: boolean, default True. False compares top losers instead of gainers.

    """    
    df_main.rename(columns={"views":"main_views"},inplace=True)
    df_comp.rename(columns={"views":"comp_views"},inplace=True)    
    df_comp = df_comp[df_comp["comp_views"] >= 100]
    comp = df_comp.merge(df_main, how="left", on=["post"])    
    comp["gain_loss"] = (comp["main_views"] / comp["comp_views"] - 1) * 100

    comp = comp.sort_values(by="gain_loss", ascending = not gain).reset_index().drop("index",axis=1)
    comp = comp.iloc[0:10,:]
    
    comp.rename(columns={"post":"Post"
                         , "comp_views": f"{comp_name} Views"
                         , "main_views": "Yesterday Views"
                         , "gain_loss": "Gain (Loss) %"},inplace=True)
    comp["Gain (Loss) %"]=comp["Gain (Loss) %"].map('{:,.0f}%'.format)
    comp[f"Yesterday Views"]=comp[f"Yesterday Views"].map('{:,.0f}'.format)
    comp[f"{comp_name} Views"]=comp[f"{comp_name} Views"].map('{:,.0f}'.format)
    
     # create chart
    fig_movers = go.Figure(
        data=[go.Table(
        columnwidth = [40,18,18,18],
        header=dict(values=list(comp.columns),
                    fill_color='#00383E',
                    font_color="white",
                    align='left'),
        cells=dict(values=comp.iloc[:,0:].T,
                    fill_color= "#F9F8F5",
                   # format = [None, ",.2f",None,None],
                    # height = 20,
                    align='left'))
        ])
    
    if gain:
        title = "Top Gainers"
    else:
        title = "Top Losers"
    fig_movers.layout.update(height=500, width = 1000, title = title)
    
    st.plotly_chart(fig_movers)   
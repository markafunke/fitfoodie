#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to evaluate time series predictions at different cutoffs
This file was used as an iterative process to test best fitting models that
are ultimately used in app.py

@author: markafunke
"""

import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

def cross_val_cutoffs(df, cutoff_dates, mode, months = 1):
    """
    Fits models and forecasts results for each cutoff date in cutoff_dates.
    Fit period is every day prior to cutoff date.
    Forecast period is months(input) after cutoff date.
    Used as rolling window cross validation to test model parameters.
    Returns dataframe of projections for each cutoff date.

    Parameters
    ----------
    df : dataframe with target variable named "y", and datetime column named "ds"
    cutoff_dates : list of datetime objects to test model on
    mode : string, "additive" or "multiplicative", sets Prophet seasonality mode
    months : integer, number of months to forecast out at each cutoff date
                default is 1 month
    """
    forecast_df = pd.DataFrame()
    for cutoff in cutoff_dates:
        
        # fit model on just dates before each cutoff
        df["cap"] = 45
        df["floor"] = 0
        
        fit_cutoff = df["ds"] < cutoff
        df_fit = df[fit_cutoff]

        model = Prophet(
            growth = "linear",
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            holidays=holidays,
            seasonality_mode = "multiplicative",
            # seasonality_prior_scale = 20
            ).add_seasonality(
                name="yearly",
                period = 365.25,
                fourier_order = 10,
                mode = "multiplicative"
            ).add_seasonality(
                name = "weekly",
                period = 7,
                fourier_order = 7,
                mode = "multiplicative"
            ).add_seasonality(
                name = "quarterly",
                period = 365.25/4,
                fourier_order = 3,
                mode = "multiplicative"
            )    
        model.add_country_holidays(country_name='US')
        model.fit(df_fit)
        
        # predict model on 1 month, 3 months, 6 months, or 12 months post cutoff
        pred_cutoff = (df["ds"] >= cutoff) & (df["ds"] < cutoff + pd.offsets.MonthBegin(months))
        df_pred = df[pred_cutoff].reset_index().drop("index",axis=1)
        df_pred["cap"] = 45
        df_pred["floor"] = 0
        

        forecast = model.predict(df_pred[["ds","cap","floor"]])
        forecast["y"] = df_pred["y"]
        forecast["cutoff"] = cutoff
        forecast_df = pd.concat([forecast_df,forecast])
    return forecast_df

def metrics(df,rpm=False):
    """
    Intended to receive output from cross_val_cutoffs() and calculate
    mean absolute error, mean absolute percent error, and median absolute
    percent error for a set of cutoff dates.

    Parameters
    ----------
    df : dataframe, output from cross_val_cutoffs()
    rpm : boolean, True if testing RPM, false if testing Views or Earnings.
        default is False.

    """
    if rpm:
        df = df.groupby("cutoff")["yhat","y"].mean()
    else:
        df = df.groupby("cutoff")["yhat","y"].sum()
    df["ae"] = np.abs(df['y'] - df['yhat'])
    df["ape"] = np.abs(df["ae"] / df['y'])
    
    print(f"mean abs error: {round(np.mean(df.ae),2)}")
    print(f"mean perc error: {round(np.mean(df.ape)*100,2)}%")
    print(f"median perc error: {round(np.median(df.ape)*100,2)}%")
    
    return df

def plot_cross_val(df):
    """
    Plots cross validation results from cross_val_cutoffs() function output
    Compares forecasted to actual for each cutoff period.

    """
    df = df.groupby("cutoff")["yhat","y"].sum()
    plt.scatter(df.index,df.y)
    plt.scatter(df.index,df.yhat)
    plt.plot(df.index,df.y)
    plt.plot(df.index,df.yhat);

# Load in data, adjust column names to fit prophet specifications
df = pd.read_csv('data/Total_Views_20150101-20200831.csv')
df.rename(columns={"Day Index": "ds", "Pageviews": "y"}, inplace=True)
df.ds = pd.to_datetime(df.ds)

df_rpm = pd.read_csv('data/Earnings_2017-05-01_2020-08-31.csv')
df_rpm.rename(columns={"Start Date": "ds", "RPM": "y"}, inplace=True)
df_rpm.ds = pd.to_datetime(df_rpm.ds)

holidays = pd.read_csv('data/holidays.csv')

# Test and plot various models and cutoff dates
cutoff_dates = pd.date_range('2017-01-01','2020-09-01', freq='1M')-pd.offsets.MonthBegin(1)
forecast_df = cross_val_cutoffs(df,cutoff_dates, mode = "multiplicative")
mets = metrics(forecast_df)
plot_cross_val(forecast_df)

cutoff_dates = pd.date_range('2018-06-01','2020-03-01', freq='1M')-pd.offsets.MonthBegin(1)
forecast_df_rpm = cross_val_cutoffs(df_rpm,cutoff_dates, mode = "multiplicative")
mets_rpm = metrics(forecast_df_rpm, rpm=True)
plot_cross_val(forecast_df_rpm)



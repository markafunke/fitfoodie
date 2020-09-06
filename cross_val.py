#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to evaluate time series predictions at different cutoffs

@author: markfunke
"""

import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt

def cross_val_cutoffs(df, cutoff_dates, mode, months = 1):
    forecast_df = pd.DataFrame()
    for cutoff in cutoff_dates:
        
        # fit model on just dates before each cutoff
        fit_cutoff = df["ds"] < cutoff
        df_fit = df[fit_cutoff]
        model = Prophet(holidays=holidays,seasonality_mode=mode)
        model.fit(df_fit)
        
        # predict model on 1 month, 3 months, 6 months, or 12 months post cutoff
        pred_cutoff = (df["ds"] >= cutoff) & (df["ds"] < cutoff + pd.offsets.MonthBegin(months))
        df_pred = df[pred_cutoff].reset_index().drop("index",axis=1)
        

        forecast = model.predict(df_pred[["ds"]])
        forecast["y"] = df_pred["y"]
        forecast["cutoff"] = cutoff
        forecast_df = pd.concat([forecast_df,forecast])
    return forecast_df


def metrics(df):
    df = df.groupby("cutoff")["yhat","y"].sum()
    df["ae"] = np.abs(df['y'] - df['yhat'])
    df["ape"] = np.abs(df["ae"] / df['y'])
    
    print(f"mean abs error: {round(np.mean(df.ae),-1)}")
    print(f"mean perc error: {round(np.mean(df.ape)*100,2)}%")
    print(f"median perc error: {round(np.median(df.ape)*100,2)}%")
    
    return df


def plot_cross_val(df):
    df = df.groupby("cutoff")["yhat","y"].sum()
    plt.scatter(df.index,df.y)
    plt.scatter(df.index,df.yhat)
    plt.plot(df.index,df.y)
    plt.plot(df.index,df.yhat);
    


# Load in data, adjust column names to fit prophet specifications
df = pd.read_csv('data/Total_Views_20150101-20200831.csv')
df.rename(columns={"Day Index": "ds", "Pageviews": "y"}, inplace=True)
df.ds = pd.to_datetime(df.ds)

# COVID caused unusually high March views that can't be expected as a normal March seasonality
covid = pd.DataFrame({
  'holiday': 'covid',
  'ds': pd.date_range('2020-03-18','2020-06-05', freq='1D'),
  'lower_window': 0,
  'upper_window': 1,
})

holidays = covid

cutoff_dates = pd.date_range('2017-01-01','2020-09-01', freq='1M')-pd.offsets.MonthBegin(1)

forecast_df1 = cross_val_cutoffs(df,cutoff_dates, mode = "additive")
forecast_df2 = cross_val_cutoffs(df,cutoff_dates, mode = "multiplicative")
additive = metrics(forecast_df1)
plot_cross_val(forecast_df1)
multiplicative = metrics(forecast_df2)
plot_cross_val(forecast_df2)

cutoff_dates = pd.date_range('2017-01-01','2020-01-01', freq='1Y')-pd.offsets.YearBegin(1)
forecast_df3 = cross_val_cutoffs(df,cutoff_dates, mode = "additive", months = 12)
forecast_df4 = cross_val_cutoffs(df,cutoff_dates, mode = "multiplicative", months = 12)
additive2 = metrics(forecast_df3)
plot_cross_val(forecast_df3)
multiplicative2 = metrics(forecast_df4)
plot_cross_val(forecast_df4)

#Tasks
# Project RPM
# Combine the forecasts
# Build app -- MVP DONE!
# Fine tune models
# Add API functionality to app


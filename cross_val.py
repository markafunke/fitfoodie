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
test1 = forecast_df1.groupby("cutoff")["yhat","y"].sum()

plt.scatter(test1.index,test1.y)
plt.scatter(test1.index,test1.yhat)
plt.plot(test1.index,test1.y)
plt.plot(test1.index,test1.yhat)
;

forecast_df2 = cross_val_cutoffs(df,cutoff_dates, mode = "multiplicative")
test2 = forecast_df2.groupby("cutoff")["yhat","y"].sum()

plt.plot(test2.index,test2.y)
plt.plot(test2.index,test2.yhat)
plt.scatter(test2.index,test2.y)
plt.scatter(test2.index,test2.yhat)
;



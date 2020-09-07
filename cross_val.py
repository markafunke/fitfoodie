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


def metrics(df,rpm=False):
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
    df = df.groupby("cutoff")["yhat","y"].sum()
    plt.scatter(df.index,df.y)
    plt.scatter(df.index,df.yhat)
    plt.plot(df.index,df.y)
    plt.plot(df.index,df.yhat);
    

model_views = Prophet(holidays=holidays,seasonality_mode="multiplicative")
model_views.fit(df)
future_views = model_views.make_future_dataframe(periods=365)
forecast_views = model_views.predict(future_views)

plot_views = model_views.plot(forecast_views)
plot_views_comp = model_views.plot_components(forecast_views)

model_rpm = Prophet(holidays=holidays,seasonality_mode="multiplicative")
model_rpm.fit(df_rpm)
future_rpm = model_rpm.make_future_dataframe(periods=365)
forecast_rpm = model_rpm.predict(future_rpm)
forecast_rpm = forecast_rpm[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
plot_rpm = model_rpm.plot(forecast_rpm)
plot_rpm_comp = model_rpm.plot_components(forecast_rpm)

# merge views and rpm together for forecast
views_merge = forecast_views[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
rpm_merge = forecast_rpm[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

views_merge.rename(columns={"yhat": "yhat_v", "yhat_lower": "yhat_lower_v", "yhat_upper": "yhat_upper_v"}, inplace=True)

earnings_df = rpm_merge.merge(views_merge, how="left", on=["ds"])    
earnings_df["rev"] = earnings_df["yhat_v"]/1000 * earnings_df["yhat"]
earnings_df["rev_l"] = earnings_df["yhat_lower_v"]/1000 * earnings_df["yhat_lower"]
earnings_df["rev_h"] = earnings_df["yhat_upper_v"]/1000 * earnings_df["yhat_upper"]

earnings_df = earnings_df.merge(df, how="left", on=["ds"])
earnings_df = earnings_df.merge(df_rpm, how="left", on=["ds"])
earnings_df = earnings_df.rename(columns={"y_x":"y_view", "y_y":"y_rpm"})
earnings_df["rev_actual"] = earnings_df["y_view"]/1000 * earnings_df["y_rpm"]

plt.scatter(earnings_df.ds,earnings_df.rev_actual)
plt.scatter(earnings_df.ds,earnings_df.rev);
plt.plot(earnings_df.ds,earnings_df.rev_actual)
plt.plot(earnings_df.ds,earnings_df.rev);


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

df_rpm = pd.read_csv('data/Earnings_2017-05-01_2020-08-31.csv')
df_rpm.rename(columns={"Start Date": "ds", "RPM": "y"}, inplace=True)
df_rpm.ds = pd.to_datetime(df_rpm.ds)

cutoff_dates = pd.date_range('2019-01-01','2020-09-01', freq='1M')-pd.offsets.MonthBegin(1)
rpm_cv_add = cross_val_cutoffs(df_rpm,cutoff_dates, mode = "additive")
rpm_cv_mult = cross_val_cutoffs(df_rpm,cutoff_dates, mode = "multiplicative")
additive_rpm = metrics(rpm_cv_add,rpm=True)
plot_cross_val(rpm_cv_add)
mult_rpm = metrics(rpm_cv_mult,rpm=True)
plot_cross_val(rpm_cv_mult)


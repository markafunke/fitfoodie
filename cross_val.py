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

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            holidays=holidays,
            seasonality_mode=mode
            ).add_seasonality(
                name="yearly",
                period = 365.25,
                fourier_order = 10
            ).add_seasonality(
                name = "weekly",
                period = 7,
                fourier_order = 7
            )
        model.add_country_holidays(country_name='US')
        model.fit(df_fit)
        
        # predict model on 1 month, 3 months, 6 months, or 12 months post cutoff
        pred_cutoff = (df["ds"] >= cutoff) & (df["ds"] < cutoff + pd.offsets.MonthBegin(months))
        df_pred = df[pred_cutoff].reset_index().drop("index",axis=1)
        

        forecast = model.predict(df_pred[["ds"]])
        forecast["y"] = df_pred["y"]
        forecast["cutoff"] = cutoff
        forecast_df = pd.concat([forecast_df,forecast])
    return forecast_df


forecast_df2 = cross_val_cutoffs(df,cutoff_dates, mode = "multiplicative")
multiplicative = metrics(forecast_df2)
plot_cross_val(forecast_df2)



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
    

model_views = Prophet(
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
            )
  
model_views.add_country_holidays(country_name='US')
model_views.train_holiday_names
model_views.fit(df)
future_views = model_views.make_future_dataframe(periods=365)
forecast_views = model_views.predict(future_views)
plot_views_comp = model_views.plot_components(forecast_views)

help(Prophet.plot_components)

plot_views = model_views.plot(forecast_views)
plot_views_comp = model_views.plot_components(forecast_views)

model_rpm = Prophet(holidays=holidays,seasonality_mode="multiplicative")
model_rpm.fit(df_rpm)
future_rpm = model_rpm.make_future_dataframe(periods=365)
forecast_rpm = model_rpm.predict(future_rpm)
plot_rpm = model_rpm.plot(forecast_rpm)
plot_rpm_comp = model_rpm.plot_components(forecast_rpm)

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

# Plot vs same week last year
newest_row = df.dropna().sort_values("ds",ascending=False).iloc[1]
current_day = newest_row["ds"]
current_week = newest_row["week"]
current_year = newest_row["year"]

last_week_day = current_day - datetime.timedelta(weeks=1)
last_year_day_high = current_day - datetime.timedelta(weeks=52)
last_year_day_low = current_day - datetime.timedelta(weeks=53)

last_yr_wk = df[(df["ds"] > last_year_day_low) & (df["ds"] <= last_year_day_high)].reset_index()
last_wk = df[(df["ds"] > last_week_day) & (df["ds"] <= current_day)].reset_index()

plt.scatter(last_yr_wk.index,last_yr_wk.views_true)
plt.scatter(last_wk.index,last_wk.views)
plt.plot(last_yr_wk.index,last_yr_wk.views)
plt.plot(last_wk.index,last_wk.views)
plt.xticks(ticks=range(7),labels=last_wk.day_of_week);

# Plot vs last week




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

# Include holidays and holiday weekends
historic_holidays =

# holidays_prior_scale (20-40), higher more impact

# n_changepoints





holidays = covid
cutoff_dates = pd.date_range('2017-01-01','2020-09-01', freq='1M')-pd.offsets.MonthBegin(1)
forecast_df1 = cross_val_cutoffs(df,cutoff_dates, mode = "additive")
additive = metrics(forecast_df1)
plot_cross_val(forecast_df1)


cutoff_dates = pd.date_range('2017-01-01','2020-01-01', freq='1Y')-pd.offsets.YearBegin(1)
forecast_df3 = cross_val_cutoffs(df,cutoff_dates, mode = "additive", months = 12)
forecast_df4 = cross_val_cutoffs(df,cutoff_dates, mode = "multiplicative", months = 12)
additive2 = metrics(forecast_df3)
plot_cross_val(forecast_df3)
multiplicative2 = metrics(forecast_df4)
plot_cross_val(forecast_df4)


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


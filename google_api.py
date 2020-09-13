#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 13:42:52 2020

@author: markfunke
"""


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
top["Day Rank"] = range(1,11)


ly_df = ly_df[ly_df["ga:pageviews"] >= 100]
comp = ly_df.merge(views_merge, how="left", on=["ds"])    



# 1 - top posts yesterday, top posts last year
# 2 - biggest winners (min. 100 views)
# 3 - biggest losers (min. 100 views)
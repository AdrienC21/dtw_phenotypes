"""
!pip install -U xlrd  # upgrade excel package to load specific files
# county choropleth graph
!pip install -U geopandas
!pip install -U pyshp
!pip install -U shapely
!pip install -U plotly-geo
!pip install -U xgboost
!pip install -U lightgbm
!pip install -U scikit-learn
!pip install -U tslearn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob
import math
import pickle
import datetime


def retrieve_covid_timeseries(folder):
    if os.path.exists(os.path.join(folder, "COVID-19_timeseries.csv")):
        df = pd.read_csv(os.path.join(folder, "COVID-19_timeseries.csv"))
        return df
    df_to_concatenate = []
    for filename in glob.glob(f"{folder}/COVID-19_JHU_data/csse_covid_19_daily_reports/*.csv"):
        df_date = pd.read_csv(filename)
        if "FIPS" in df_date.columns:  # county-level data available
            df_date = df_date[df_date["Country_Region"] == "US"]
            date = filename.split("/")[-1][:-4]
            date = date[6:] + "-" + date[:2] + "-" + date[3:5]
            df_date.index = df_date["FIPS"].apply(lambda x: str(int(x)) if not(math.isnan(x)) else np.nan)
            df_date = df_date[["FIPS", "Deaths"]].dropna()  # remove county with no FIPS/no data
            df_date = df_date[["Deaths"]]
            df_date.columns = [date]
            df_date = df_date.T

            # Merge duplicated counties
            if df_date.columns.duplicated().any():
                df_date = df_date.sum(axis=1, level=0, skipna=True)

            df_to_concatenate.append(df_date)

    # Merge all the dates
    df = pd.concat(df_to_concatenate)
    # Sort the dataframe by chronological order
    df.sort_index(inplace=True)
    df = df.fillna(method="ffill")  # fill na forward
    df = df.cummax()  # cumulative max to deal with sudden drop of deaths
    df = df.fillna(0)  # then, nan=no data before (so replace with 0)

    # applied a 7-day average to smooth the daily death counts to account for noise
    # in the data at the daily level
    df = df.rolling(window=7).mean()

    # save results
    df.to_csv(os.path.join(folder, "COVID-19_timeseries.csv"), index=False)
    return df


def z_score_normalize(df):
    A = df.fillna(0)  # we can assume that no data = no death
    A_zscore = (A - A.mean(axis=0)) / A.std(axis=0)

    # Drop counties with no values (only nan and or 0 death)
    county_to_drop = []
    for fips in list(A_zscore.columns):
        if np.isnan(A_zscore[fips]).sum() == len(A_zscore):
            county_to_drop.append(fips)
    print("{} counties dropped".format(len(county_to_drop)))
    A_zscore.drop(columns=county_to_drop, inplace=True)
    A_zscore = A_zscore.T  # transpose the matrix: for clustering
    return A_zscore



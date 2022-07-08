from utils import retrieve_covid_timeseries, z_score_normalize
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from time import perf_counter
import numpy as np
from threading import Thread, Lock

# folder = "../temporal_analysis_us_county"
folder = ""

df_covid = retrieve_covid_timeseries(folder)
if "Unnamed: 0" in df_covid.columns:
    df_covid.index = df_covid["Unnamed: 0"]
    df_covid.drop(columns=["Unnamed: 0"], inplace=True)

df_covid.index = pd.to_datetime(df_covid.index)
A_zscore = z_score_normalize(df_covid)
A_zscore_array = A_zscore.values

"""
top = perf_counter()
distances = np.zeros((20, 20))
for i in range(20):
    for j in range(i+1, 20):
        distance, _ = fastdtw(A_zscore_array[i][:100], A_zscore_array[j][:100], dist=euclidean)
        distances[i, j] = distance
        distances[j, i] = distance

print(distances)
print(perf_counter() - top)
"""


def record_distance(i, j, distance, distances, lock):
    lock.acquire()
    distances[i, j] = distance
    distances[j, i] = distance
    lock.release()


def calc_dist(i, j):
    global distances, lock
    distance, _ = fastdtw(A_zscore_array[i][:100], A_zscore_array[j][:100], dist=euclidean)
    record_distance(i, j, distance, distances, lock)


top = perf_counter()
lock = Lock()
distances = np.zeros((20, 20))
threads = []
for i in range(20):
    for j in range(i+1, 20):
        t = Thread(target=calc_dist, args=(i, j))
        threads.append(t)
for t in threads:
    t.start()
for t in threads:
    t.join()
print(distances)
print(perf_counter() - top)

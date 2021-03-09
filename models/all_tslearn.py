import numpy as np
import pandas as pd
import matplotlib
from importlib import reload
import matplotlib.pyplot as plt
import elements
elements = reload(elements)
from elements.event import Event
import os
import tslearn
from tslearn.utils import to_time_series_dataset
from tslearn.utils import to_time_series
from tslearn.clustering import KernelKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
#%%
#read the main dataset
true_clusters_known = pd.read_pickle('data/known_true_clusters_ids.pkl')
all_clusters_data = []
cl = 'Tree'
# min_size = 1280
# for
# for ev in true_clusters_known[cl].dropna():
#     e = Event(ev, 0, -1, 'resampled').data.shape[0]
#     if min_size > e:
#         min_size = e
# print(e)

for ev in true_clusters_known[cl].dropna():
    e = Event(ev, 0, -1, 'resampled')
    selected_data = e.res().loc[:, e.data.columns != 'Time (s)']
    all_clusters_data.append(selected_data)
    

#%%
seed = 0
formatted_dataset = to_time_series_dataset(all_clusters_data)
formatted_dataset[np.isnan(formatted_dataset)] = 0
#%%
X_train = TimeSeriesScalerMeanVariance().fit_transform(formatted_dataset)
gak_km = KernelKMeans(n_clusters=3,
                      kernel="gak",
                      kernel_params={"sigma": "auto"},
                      n_init=20,
                      verbose=True,
                      random_state=seed)
y_pred = gak_km.fit_predict(X_train)
#%%




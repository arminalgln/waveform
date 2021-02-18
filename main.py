import numpy as np
import pandas as pd
import matplotlib
from importlib import reload
import matplotlib.pyplot as plt
import elements
elements = reload(elements)
from elements.event import Event
import os
from scipy.fft import fft, fftfreq

#%%
#meta data
meta_event = pd.read_csv('data/meta_data.csv')
#List of events
plt.ion()
whole_events = [i.split('.')[0] for i in os.listdir('data/csv')]
#just testing some data
id = whole_events[100]
start= 0
end = -1
e = Event(id, start, end)
keys = e.data.keys()
yf, yf_mag_real, xf, start_index, N, T = e.fft_analyzer(keys[6])
fig = e.show_detail()
print(meta_event.loc[meta_event['EventId']==int(id)].values)


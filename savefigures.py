import numpy as np
import pandas as pd
import matplotlib
from importlib import reload
import matplotlib.pyplot as plt
import elements
elements = reload(elements)
from elements.event import Event
import os
from scipy.fft import fft, fftfreq, ifft

#%%
#meta data
meta_event = pd.read_csv('data/meta_data.csv')
#List of events
plt.ion()
whole_events = [i.split('.')[0] for i in os.listdir('data/csv')]
#just testing some data
id = whole_events[0]
start= 0
end = -1
e = Event(id, start, end)
keys = ['Time (s)', ' Va', ' Vb', ' Vc', ' Ia', ' Ib', ' Ic', ' In']
yf, yf_mag_real, xf, start_index, N, T = e.fft_analyzer(keys[6])
fig = e.show_detail()
print(meta_event.loc[meta_event['EventId']==int(id)].values)
#%%
#get the fft for each event as the input features
whole_events = [i.split('.')[0] for i in os.listdir('data/csv')]
number_of_freqs = 200
Nf = number_of_freqs * 7 # number of features
features = {}
max_voltage = 0
max_current = 0
bad_data_index= [122, 123, 124, 125, 127, 128]
unique_causes = meta_event['Cause'].unique()
bad_events_id = []
causes = pd.DataFrame(columns={'id','cause','label'})
#%%

#saving figures just for visual reference
causes = pd.read_pickle('data/causes.pkl')
for ev in whole_events[0:1]:
    path = "figures/eventshape/{}.png".format(ev)
    e = Event(ev, start, end)
    fig = e.show_detail()
    cause = causes.loc[causes['id'] == ev]['cause'].values[0]
    plt.title("{}".format(cause))
    plt.show()
    fig.savefig(path)
#%%
#save figures by their known group
causes = pd.read_pickle('data/causes.pkl')
causes = causes.groupby('cause')
all_causes = causes.groups.keys()
for c in all_causes:
    path = 'figures/knownclusters/{}'.format(c)
    os.mkdir(path)
    for ev in causes.get_group(c)['id']:
        e = Event(ev,0,-1)
        fig = e.show_detail()
        cause = c
        plt.title("{}".format(cause))
        plt.show()
        save_path = path + '/{}.png'.format(ev)
        fig.savefig(save_path)

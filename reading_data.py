import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import urllib.request
import elements
from elements.event import Event
import numpy as np
import math
import scipy
from scipy import interpolate
#%%

URL = 'https://pqmon.epri.com/see_all.html'
page = requests.get(URL)

#main http page
soup = BeautifulSoup(page.content, 'html.parser')
#main table with event info and link to them
table = soup.find(id='demo')
header = table.thead
body  = table.tbody
#base url for each event page
id='0001'
base_url = "https://pqmon.epri.com/doe_folder/{}/index.html".format(id)
#%%
#making key for event table dictionary
keys = []
for th in header.find_all('th'):
    keys.append(th.get_text())

#%%

#save info for each event in the main data dict
event_meta = {key: list() for key in keys}
event_data = dict()
event_text = dict()
#%%
#event meta data
for row in body.find_all('tr'):
    for idx, col in enumerate(row.find_all('td')):
        event_meta[keys[idx]].append(col.get_text())

meta_data_events = pd.DataFrame(event_meta)

meta_data_events.to_csv('data/meta_data.csv')
#%%
#get the csv data for each event

files = os.listdir('data/csv')
for idx, event in enumerate(event_meta['EventId']):
    id = event_meta['EventId'][idx].split('\n')[1]
    #print(id)
    path = 'data/csv/{}.csv'.format(id)
    if not '{}.csv'.format(id) in files:
        base_url = "https://pqmon.epri.com/doe_folder/{}/{}.csv".format(id, id)
        data = pd.read_csv(base_url, skiprows=1)
        data.to_csv(path)
#%%
#get the text file wrt each event

event_meta = pd.DataFrame(event_meta)
files = os.listdir('data/txt')
for idx, event in event_meta.iterrows():
    id = event['EventId'].split('\n')[1]
    path = 'data/txt/{}.txt'.format(id)
    if event['FeederId'].split('_')[0] == 'F':
        name = event['SiteName'] + '_' + event['FeederId']
        text_url = "https://pqmon.epri.com/doe_folder/circuits/{}.txt".format(name)
        if not '{}.txt'.format(id) in files:
            print(id)
            urllib.request.urlretrieve(text_url, path)
            print(path)
#%%
#interpolate to get the highest sampling rate
whole_events = [i.split('.')[0] for i in os.listdir('data/csv')]
cycle = 1/60
max_sampling_rate = 256 #per cycle
min_sampling_rate = 64  #per cycle


for ev in whole_events:
    e = Event(ev, 0, -1, 'csv')
    t = e.data['Time (s)'] - e.data['Time (s)'][0] #whole time window
    sampling_time = e.data['Time (s)'][1]-e.data['Time (s)'][0] #sampling time
    sample_per_cycle = math.ceil(cycle/sampling_time) #sample per cycle

    sample_horizon = np.arange(t.iloc[0], t.iloc[-1], cycle/min_sampling_rate)
    new_event_data = {'Time (s)':sample_horizon}

    for f in e.data.keys()[1:]:
        intpo = interpolate.interp1d(t,e.data[f])
        downsampled_data = intpo(sample_horizon)
        new_event_data[f] = downsampled_data
    new_event_data = pd.DataFrame(new_event_data)
    new_event_data.to_csv('data/downsampled/{}.csv'.format(ev))
    print(math.ceil((sample_horizon[1] - sample_horizon[0])/(t.iloc[1] - t.iloc[0])))



#%%
e=Event('0001',0,-1, 'resampled')
#generate noise based on the maximum change in the time series
def get_noisy(ts):
    temp = np.roll(ts,-1)
    residue = np.abs(ts[0:-1]-temp[0:-1])
    eps = max(residue)
    noise = np.random.normal(0, eps/2, ts.shape[0])
    ts_noisy = ts + noise
    return ts_noisy

ts = e.data[' In']
tn = get_noisy(ts)

plt.plot(ts)
plt.plot(tn)
plt.show()
#%%

#roll the time series
causes = pd.read_pickle('data/causes.pkl')
noise_number = 10
max_shift = 10
#data augmentation
def augmentation(ev):
    e = Event(ev, 0, -1, 'resampled')
    aug_data = e.data.copy()
    data = 
    for f in [' Va', ' Vb']:






whole_events = [i.split('.')[0] for i in os.listdir('data/csv')]

for ev_id in whole_events:


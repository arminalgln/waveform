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
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

#%%
#parsing the resource files
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
#interpolate to get the highest sampling rate then
#save all the data with the same sampling rate
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
    new_event_data.to_csv('data/resampled/{}.csv'.format(ev))
    print(math.ceil((sample_horizon[1] - sample_horizon[0])/(t.iloc[1] - t.iloc[0])))



#%%
#generate noise based on the maximum change in the time series

def get_noisy(ts):
    temp = np.roll(ts,-1)
    residue = np.abs(ts[0:-1]-temp[0:-1])
    eps = max(residue)
    #np.random.seed(0)
    noise = np.random.normal(0, eps/2, ts.shape[0])
    ts_noisy = ts + noise
    return ts_noisy
## e = Event('0001',0,-1, 'resampled')

# ts = e.data[' In']
# tn = get_noisy(ts)
#
# plt.plot(ts)
# plt.plot(tn)
# plt.show()
#%%
#data augmentation

def augmentation(ev, shift, noise_number, order):

    e = Event(ev, 0, -1, 'resampled')
    sample_horizon = e.data['Time (s)']
    indexes = np.array(list(e.data.index))
    ns = indexes.shape[0]
    cause = causes.loc[causes['id']==ev]
    augmented_causes = pd.DataFrame(columns=['label', 'id', 'cause'])
    for i, shift in enumerate(np.arange(-shift, shift + 1)):
        for n in range(noise_number):
            aug_event = {'Time (s)': sample_horizon}
            for f in e.data.keys()[1:]:
                shifted_temp_data = np.roll(e.data[f], shift)[max(0, shift): max(ns, ns - shift)]
                intpo = InterpolatedUnivariateSpline(
                    indexes[max(0, shift): max(ns, ns - shift)], shifted_temp_data, k=order
                )
                new_data = intpo(indexes)
                aug_event[f] = get_noisy(new_data)
            aug_event = pd.DataFrame(aug_event)
            new_id = cause['id'].values[0] + '_' + str(i) + '_' + str(n)
            augmented_causes = augmented_causes.append({'label': cause['label'].values[0], 'id': new_id, 'cause':cause['cause'].values[0]}, ignore_index=True)
            saving_path = 'data/augmented_data/{}.pkl'.format(new_id)
            aug_event.to_pickle(saving_path)
            print('I saved {}'.format(new_id))

    return augmented_causes


#%%
#roll the time series
causes = pd.read_pickle('data/causes.pkl')
noise_number = 10
shift = 10
order = 2
whole_events = [i.split('.')[0] for i in os.listdir('data/csv')]
bad_data_index= [122, 123, 124, 125, 127, 128]
bad_data_event = [whole_events[i] for i in bad_data_index]
whole_events = np.setdiff1d(whole_events, bad_data_event)
for i, ev in enumerate(whole_events):
    agc = augmentation(ev, shift, noise_number, order)
    if i==0:
        whole_agc =agc
    whole_agc = whole_agc.append(agc, ignore_index=True)
whole_agc.to_pickle('data/whole_agc.pkl')
#%%
#show events , rolled and noisy vs real
# I saved 0001_0_0
# I saved 0001_1_0
# I saved 0001_2_0
e = Event('4556',0,-1, 'resampled')
d = pd.read_pickle('data/augmented_data/4556_20_8.pkl')
plt.plot(d[' Ib'])
plt.plot(e.data[' Ib'])
plt.show()
#%%
#extracting just known and real dataset for evaluation
meta_event = pd.read_csv('data/meta_data.csv')
causes = pd.read_pickle('data/causes.pkl').groupby('cause')

unique_causes = meta_event['Cause'].unique()

known_clusters_cause = ['Tree', 'Equipment','Weather', 'Vehicle', 'Planned',
                        'Animal', 'Lightning', 'Customer Request', 'Customer Caused']
paper_based_known = ['Tree', 'Equipment', 'Vehicle','Animal', 'Lightning']

clusters = {}
for cl in paper_based_known:
    clusters[cl] = causes.get_group(cl)['id'].reset_index(drop=True)

clusters = pd.DataFrame(clusters)
clusters.to_pickle('data/known_true_clusters_ids.pkl')
#%%
#chnage the long and bad data to more acceptable event length with event signature
bad_events_horizon = {
    '2771':[0,500],
     '21838':[2500,3000],
     '21839':[3000,3500],
     '21844':[600,900],
     '21845':[1350,1800],
     '21846':[0,500],
     '21847':[1400,1900],
     '21848':[1600,2100],
     '21851':[3100,3400],
     '21852':[1100,1600],
     '21853':[0,500],
     '21854':[1000,1400],
     '21835':[2500,3000],
     '21836':[1400,1600],
     '21837':[800,1200],
     '21841':[100,600],
     '21850':[2000,2500],
     '21862':[800,1200],
     '21863':[500,1000],
     '21865':[3500,4000],
     '21873':[2300,2500],
     '21856':[900,1200],
     '21857':[1400,1900],
     '21858':[0,500],
     '21859':[800,1300],
     '21860':[2200,2700],
     '21861':[2300,2800],
     '21872':[1600,2000],
     '21831':[2800,3100],
     '21832':[2800,3100],
     '21833':[1300,1700],
     '21834':[1600,2100],
     '21840':[3600,3900],
     '21842':[1150,1650],
     '21843':[100,500]
             }

for ev in list(bad_events_horizon.keys()):
    e = Event(ev, 0, -1, 'resampled')
    data = e.data.iloc[bad_events_horizon[ev][0]:bad_events_horizon[ev][1]]
    data.to_csv('data/resampled/{}.csv'.format(ev))
    # plt.plot(data[' Ib'])
    # plt.plot(data[' Ic'])
    # plt.plot(data[' Ia'])
    # plt.show()

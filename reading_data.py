import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
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

#%%
files = os.listdir('data/csv')
#get the csv data for each event
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
for idx, event





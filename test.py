import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import urllib.request
import csv
from requests import get
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
#%%

base_url = "http://mis.ercot.com/misapp/GetReports.do?reportTypeId=11485&reportTitle=LMPs%20by%20Electrical%20Bus&showHTMLView=&mimicKey"
URL = "http://mis.ercot.com/"
page = requests.get(base_url)
#%%
soup = BeautifulSoup(page.content, 'html.parser')
#%%
all_titles = soup.find_all(class_="labelOptional_ind")
all_links = soup.find_all("a")
#%%
all_urls_csv = []
for idx, link in enumerate(all_links):
    if all_titles[idx].text.split('.')[-2].split('_')[-1] == 'csv':
        path = 'data/LMP/{}.csv'.format(idx)
        all_urls_csv.append(URL+link.get('href'))
        zipurl = URL+link.get('href')
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('data/LMP')
        if idx%100 == 0:
            print(idx)


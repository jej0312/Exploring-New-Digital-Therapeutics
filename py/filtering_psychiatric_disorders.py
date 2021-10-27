import pandas as pd
from bs4 import BeautifulSoup
import requests


## HTTP GET Request
req = requests.get('https://www.ncbi.nlm.nih.gov/medgen/14047#Term_Hierarchy')
html = req.text
header = req.headers

soup = BeautifulSoup(html, 'html.parser')

soup.find(attrs={'class': 'TLclosed'})

# term hierarchy for Mental disorder
diseases_ = soup.select('.TLline > a')
keys = {'cui': [], 'name': []}
for disease in diseases_:
    keys['cui'].append(disease['href'].split('/')[-1])
    keys['name'].append(disease.text)

URLs = []
for key in keys['cui']:
    url = 'https://www.ncbi.nlm.nih.gov/medgen/'+str(key)
    URLs.append(url)

# crawling
cuis = {'cui': [], 'name': []}
for url in URLs:
    try:
        req = requests.get(url)
        html = req.text
        soup = BeautifulSoup(html, 'html.parser')
        cui = soup.select('#maincontent > div > div:nth-child(5) > div > div.portlet > div > div > dl > dd:nth-child(4) > a')
        cuis['cui'].append(cui[0].text)
        name = soup.select('#maincontent > div > div:nth-child(5) > div > div.portlet > h1 > div')
        cuis['name'].append(name[0].text)
    except Exception as ex:
        print('check', url)

'''
https://www.ncbi.nlm.nih.gov/medgen/452779: Dyscalculia; C1411876
https://www.ncbi.nlm.nih.gov/medgen/137909: Mania; C0338831
'''

cuis['cui'].append('C0338831')
cuis['name'].append('Mania')

cuis['cui'].append('C1411876')
cuis['name'].append('Dyscalculia')

import pickle

with open('data/disease/psychiatric_disorder_CUIs.pickle', 'wb') as f:
    pickle.dump(cuis, f)

############################## collect list ##############################
from bs4 import BeautifulSoup
import requests


## HTTP GET Request
req = requests.get('https://www.medicalstartups.org/top/digital-therapeutics/')
html = req.text
header = req.headers

soup = BeautifulSoup(html, 'html.parser')

companies = soup.select('#content > div:nth-child(7) > div.item-title > a')

print(companies)

# open selenium driver
from selenium import webdriver
driver = webdriver.Chrome('../chromedriver')

# collect DTx company lists from https://www.medicalstartups.org/top/digital-therapeutics/
driver.get(url='https://www.medicalstartups.org/top/digital-therapeutics/')
companies = driver.find_elements_by_class_name('item-title')
for company in companies:
    print(company.text)
driver.close()


############################## check list ##############################

# ----------------- extract keys ----------------- #
import pandas as pd
import glob
import os

os.getcwd()

input_path = r"data\patent"

# file_list = glob.glob(os.path.join(input_path, "*.csv"))
file_list = glob.glob(os.path.join(input_path, "DTx*.csv"))
file_list = [file for file in file_list if file.split()[-1] != 'wfam.csv']
file_list = [file for file in file_list if file.split('\\')[-1] != 'DTx_patents_all_0719.csv']
file_list = [file for file in file_list if not(file.split('\\')[-1].startswith('G16'))]
file_list = sorted(file_list)
# file_list = file_list[7:]

list_of_df = []
for file in file_list:
    df_temp = pd.read_csv(file, encoding='cp949')
    list_of_df.append(df_temp)
    print(file.split('\\')[-1], 'appended.')
df_accum = pd.concat(list_of_df)

print(df_accum.shape)
print(df_accum.columns)

df_accum = df_accum[['WIPS ON key', 'Original IPC All']]

print(df_accum.isna().sum())

del df_temp
del list_of_df

df_accum.rename(columns={'WIPS ON key': 'wipsonkey'}, inplace=True)
df_accum.loc[:, 'wipsonkey'] = df_accum.wipsonkey.apply(lambda x: str(int(x)))

len(df_accum)


# ----------------- EDA ----------------- #

from collections import Counter

# IPC counts
Counter([ipc.strip().split('/')[0] for ipcs in df_accum['Original IPC All'] if pd.notnull(ipcs) for ipc in ipcs.split('|')])

# ----------------- update DTx ----------------- #

import sys, os
os.getcwd()
sys.path.append(os.path.abspath(os.path.join('.', 'py')))

import pymongo
import pymongo_handler
mongo = pymongo_handler.DBHandler()

db = mongo.client['patents']
collection = db['DTx']

for key in df_accum['wipsonkey']:
    collection.update_one({"wipsonkey": key}, {"$set": {"DTx": "T"}})

collection2 = db['disease_label']

cursor = collection2.find({'wipsonkey': {'$in': list(df_accum['wipsonkey'])}})

response = []
for cur in cursor:
    response.append(cur['wipsonkey'])

len(set(response))
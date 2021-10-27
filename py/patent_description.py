
'''
python 3.8.6
conda 3.9.5

github에 올린 게 더 완전함
'''

# ----------------- extract keys ----------------- #
import pandas as pd
import glob
import os

os.getcwd()

#### saved patent data from WIPSON
input_path = r"data\patent"

file_list = glob.glob(os.path.join(input_path, "*A6_*.csv"))
file_list = [file for file in file_list if file.split('\\')[-1] != 'A6_0_2400.csv']
file_list = sorted(file_list)

list_of_df = []
for file in file_list:
    df_temp = pd.read_csv(file, encoding='cp949')
    list_of_df.append(df_temp)
    print(file.split('\\')[-1], 'appended.')
df_accum = pd.concat(list_of_df)

print(df_accum.shape)
print(df_accum.columns)

# extract the columns I will use
df_accum = df_accum[['국가코드', '특허/실용 구분', '문헌종류 코드', '발명의 명칭', '요약',
       '대표청구항', '청구항 수', '출원번호', '출원일',
       '공개일', '등록번호', '등록일', '출원인',
       '출원인 국적', '출원인 수', '발명자/고안자',
       '발명자 수', '국제 출원번호',
       '국제 출원일', '국제 공개번호', '국제 공개일',
       'Original CPC Main', 'Original CPC All', 'Original IPC Main',
       'Original IPC All', '인용 문헌 수(B1)', '인용 문헌 수(F1)',
       '인용 문헌번호 (B1) + 심사관(E) 인용 (KR,US)', '인용 문헌번호 (F1) + 심사관(E) 인용 (KR,US)',
                     # B1: 해당 특허가 인용한 문헌, F1: 해당 특허를 인용한 문헌, 1: 한 단계
       '패밀리 ID', '패밀리 문헌번호 (출원기준)', '패밀리 문헌 수 (출원기준)',
       'WIPS ON key']]

print(df_accum.isna().sum())

del df_temp
del list_of_df

df_accum.rename(columns={'WIPS ON key': 'wipsonkey'}, inplace=True)
df_accum.loc[:, 'wipsonkey'] = df_accum.wipsonkey.apply(lambda x: str(int(x)))

from datetime import datetime
df_accum['updated'] = datetime.today().strftime("%Y%m%d")

# ---------------------------------- #

from collections import Counter

# IPC
Counter([str(len(ipcs.split('|'))) for ipcs in df_accum['Original IPC All'] if pd.notnull(ipcs)])
Counter([ipc.strip().split('/')[0] for ipcs in df_accum['Original IPC All'] if pd.notnull(ipcs) for ipc in ipcs.split('|')])

# Citation
Counter([str(cit) for cit in df_accum['인용 문헌 수(B1)']])
Counter([str(cit) for cit in df_accum['인용 문헌 수(F1)']])
Counter([str(key) for key in df_accum['wipsonkey']])

df_accum['인용 문헌 수(B1)'].sum()
df_accum['인용 문헌 수(F1)'].sum()


# ----------------- pymongo ----------------- #
# Bring your packages onto the path
import sys, os
os.getcwd()
sys.path.append(os.path.abspath(os.path.join('.', 'py')))

import pymongo
import pymongo_handler
mongo = pymongo_handler.DBHandler()

mongo.insert_item_many(df_accum.to_dict('records'), 'patents', 'DTx')

db = mongo.client['patents']
collection = db['DTx']

# remove duplicates
cursor = collection.aggregate(
    [
        {"$match": {"DTx": {"$ne": "T"}}}, # T는 냅두기
        {"$group": {"_id": "$wipsonkey", "unique_ids": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
        {"$match": {"count": { "$gte": 2 }}}
    ]
)

response = []
for doc in cursor:
    del doc["unique_ids"][0]
    for id in doc["unique_ids"]:
        response.append(id)

len(response)

collection.delete_many({'_id': {"$in": response}})


# print
my_query = {"출원인": "Akili Interactive Labs, Inc."}

sort = [("출원일", pymongo.ASCENDING), ("청구항수", pymongo.DESCENDING)]
skip = 0
limit = 50

doc_count = collection.count_documents(my_query, skip=skip)
docs = collection.find(my_query).sort(sort).skip(skip).limit(limit)
print('total counts:', doc_count)
for doc in docs:
    print(doc['출원일'], doc['wipsonkey'])


# ----------------- crawling description ----------------- #
import sys, os
os.getcwd()
sys.path.append(os.path.abspath(os.path.join('.', 'py')))

import pymongo
import pymongo_handler
mongo = pymongo_handler.DBHandler()

# 제대로 들어갔는지 확인
db = mongo.client['patents']
collection = db['DTx']

def check_exists_by_xpath(xpath):
    from selenium.common.exceptions import NoSuchElementException
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True
 

# extract the list of patents with no description
from datetime import datetime
cursor = collection.find({'description': {'$exists': False }, # None,
                          'updated': '20210719', # datetime.today().strftime("%Y%m%d")
                          }, {'wipsonkey':1, '_id':0})
docs = []
for doc in cursor:
    print(doc.values())
    docs.append(doc['wipsonkey'])

len(docs)


# get description from WIPSON
## setting
from selenium import webdriver
driver = webdriver.Chrome('../chromedriver')
url = 'https://www.wipson.com/service/mai/main.wips'
driver.get(url=url)

main = driver.window_handles
for handle in main:
    if handle != main[0]:
        driver.switch_to.window(handle)
        driver.close()
driver.switch_to.window(main[0])

# login
driver.find_element_by_id('username').send_keys('##########') # WIPSON ID
driver.find_element_by_id('password').send_keys('##########') # WIPSON PW
driver.find_element_by_xpath('//*[@id="devLoginInputArea"]/a').click()

from selenium.webdriver.common.alert import Alert
Alert(driver).accept()

import time, random
driver.implicitly_wait(time_to_wait=random.randint(9, 10))

main = driver.window_handles
for handle in main:
    if handle != main[0]:
        driver.switch_to.window(handle)
        driver.close()
driver.switch_to.window(main[0])


# get the description of patents
from tqdm import tqdm
for i, doc in tqdm(enumerate(docs)):
    try:
        key = doc.strip()

        # search
        url = 'https://www.wipson.com/service/doc/docView.wips?skey='+str(key)
        driver.get(url)
        driver.implicitly_wait(time_to_wait=random.randint(2, 5))

        driver.find_element_by_xpath('//*[@id="devTabBlock"]/ul/li[3]/a').click()

        timesleep = random.randint(2, 5)
        time.sleep(timesleep)

        driver.implicitly_wait(time_to_wait=random.randint(2, 5))
        description = driver.find_element_by_xpath('//*[@id="devInclude"]/div[2]/table/tbody/tr/td[1]/div').text

        collection.update_one({"wipsonkey": key}, {"$set": {"description": description}})

    except Exception as ex:
        print(key, "("+str(i)+"): ", ex)

driver.close()


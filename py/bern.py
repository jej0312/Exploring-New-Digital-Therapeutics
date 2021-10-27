
# ----------------- description bern ----------------- #
import sys, os
os.getcwd()
sys.path.append(os.path.abspath(os.path.join('.', 'py')))

import pymongo
import pymongo_handler
mongo = pymongo_handler.DBHandler()

db = mongo.client['patents']
collection = db['DTx']

from datetime import datetime

# extract the disease names from description of patents
## get the list of description
cursor = collection.find({'updated': datetime.today().strftime("%Y%m%d")}, {'wipsonkey': 1, 'description':1, '_id':0})
cursor = collection.find({
        'DTx': {
            '$in': [
                'C', 'T'
            ]
        },
        '출원일': {
            '$gte': '2000-01-01',
            '$lte': '2021-06-31'
        },
        'updated': '20210719'
    }, {'wipsonkey': 1, 'description':1, '_id':0})

docs = {'wipsonkey': [], 'description': []}
for doc in cursor:
    try:
        docs['description'].append(doc['description'])
        docs['wipsonkey'].append(doc['wipsonkey'])
    except Exception as ex:
        print(doc['wipsonkey'], 'has no description.')

len(docs['wipsonkey'])

import os
import requests
import re
from tqdm import tqdm
import pickle

def query_raw(text, url="https://bern.korea.ac.kr/plain"):
    return requests.post(url, data={'sample_text': text}).json()

def disease_id(id_list, key):
    diseases = {'wipsonkey': [], 'BERN': [], 'MESH': [], 'OMIM': [], 'CUI-less': []}
    for disease in id_list:
        diseases['wipsonkey'] = key
        if disease.startswith('BERN'):
            diseases['BERN'] = disease.split('BERN:')[-1]
        if disease.startswith('MESH'):
            diseases['MESH'] = disease.split('MESH:')[-1]
        if disease.startswith('OMIM'):
            diseases['OMIM'] = disease.split('OMIM:')[-1]
        if disease.startswith('CUI-less'):
            diseases['CUI-less'] = True
    return diseases


error_list = {'wipsonkey': [], 'paragraph': []}

## to extract the patents with no target disorders yet
collection2 = db['disease_label']
cursor = collection2.find({}, {'wipsonkey': 1, 'description':1, '_id':0})
tmp = []
for c in cursor:
    tmp.append(c['wipsonkey'])
len(tmp)

def extract(x):
    return False if x in tmp else True

tmp = list(filter(extract, docs['wipsonkey'])) # the patents to apply BERN

## extract disease names from description, and update it on DB
if __name__ == '__main__':
    start = 0
    for i, doc in tqdm(enumerate(docs['description'][start:])):
        if docs['wipsonkey'][i+start] in tmp:
            try:
                paragraph_ = re.split('\[[0-9]*\]', doc)
                for paragraph in paragraph_: #
                    if paragraph != '':
                        raw = query_raw(str(paragraph))
                        if len(raw['denotations']) != 0:
                            for value in raw['denotations']:
                                if value['obj'] == 'disease':
                                    diseases = disease_id(value['id'], docs['wipsonkey'][i+start])
                                    mongo.insert_item_one(diseases, 'patents', 'disease_label')

            except Exception as er:
                error_list['wipsonkey'].append(docs['wipsonkey'][i+start])
                error_list['paragraph'].append(str(paragraph))

                with open('data/error_list_diseaselabel.pickle', 'wb') as f:
                    pickle.dump(error_list, f, pickle.HIGHEST_PROTOCOL)


# remove duplicates
cursor = collection2.aggregate(
    [
        {"$group": {"_id": {"key": "$wipsonkey", "bernid": "$BERN", "CUI": "$CUI-less"}, "count": {"$sum": 1}, "docs": {"$push": "$_id"}}},
        {"$match": {"count": { "$gte": 2 }}}
    ]
)

response = []
for doc in cursor:
    # print(doc['docs'][1:])
    response.append(doc['docs'][1:])

response = sum(response, [])
len(response)

collection2.delete_many({'_id': {"$in": response}})


# keep only diseases with MESH ID
cursor = collection2.find({
    'MESH': {
        '$size': 0
    }
}, {'_id': 1})

response = []
for doc in cursor:
    # print(doc)
    response.append(doc['_id'])
len(response)

collection2.delete_many({'_id': {"$in": response}})
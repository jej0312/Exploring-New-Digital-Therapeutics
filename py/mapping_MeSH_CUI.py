# https://meshb.nlm.nih.gov/record/ui?ui=D006816 (MeSH key)

################ mapping file ################

import zipfile

zip = zipfile.ZipFile('data/umls-2021AA-mrconso.zip')
zip.extract('MRCONSO.RRF', 'data')

zip.close()


'''
https://towardsdatascience.com/use-embeddings-to-predict-therapeutic-area-of-clinical-studies-654af661b949
https://github.com/thierryherrmann/simple-machine-learning/blob/master/2018-11-10-studies-therapeutic-areas/studies-therapeutic-areas.ipynb

!cut -d'|' -f1,12,14,15 ~/data/mapping/MRCONSO.RRF/MRCONSO.RRF > ~/data/mapping/MRCONSO.RRF/MRCONSO_reduced.RRF
'''

import pandas as pd

def df_mem(df):
    return '%.1f Mb' % (df.memory_usage(index=True, deep=True).values.sum() / 1024 / 1024)

# since the file is big, need a special function to read in streaming mode and eliminate on the fly
# CUIs that are not in the embeddings
def load_conso(file_name, keys):
    rows = []
    cnt = 0
    with open(file_name, encoding="ISO-8859-1") as fp:
        for cnt, line in enumerate(fp):
            line = line.strip()
            cols = line.split('|')
            cols[3] = cols[3].lower()
            if cols[1] in keys:
                rows.append(cols)
            cnt += 1

    # print("loaded '%s', %d rows (%s)" % (file_name, len(df), df_mem(df)))
    df = pd.DataFrame(rows, columns=['CUI', 'SAB', 'CODE', 'STR'])
    print("loaded '%s', %d rows (%s)" % (file_name, len(df), df_mem(df)))
    print('processed rows: %d' % cnt)
    return df

# load data
df_c = load_conso('data/mapping/MRCONSO_reduced.RRF', ['MSH'])

################ CUIs to search ################
# Bring your packages onto the path
import sys, os
os.getcwd()
sys.path.append(os.path.abspath(os.path.join('.', 'py')))

import pymongo
import pymongo_handler
mongo = pymongo_handler.DBHandler()
db = mongo.client['patents']
collection = db['disease_label']

docs = collection.find({ 'MESH' : { '$exists': True, '$ne': [] }, 'CUI_list': { '$exists': False } })
meshid = []
for doc in docs:
    meshid.append(doc['MESH'])

vocab_keys = list(set(meshid))


################ upload on DB after mapping ################
# mapping IDs
from collections import defaultdict
mesh_code_to_cui = defaultdict(set) # used to link a study to CUIs (through mesh codes)
                                    # used later to associate therapeutic area (from their strings) to CUIs

for row in df_c[df_c.SAB=='MSH'][['CODE', 'CUI']].itertuples():
    code, cui = row[1], row[2]
    if code in vocab_keys:
        mesh_code_to_cui[code].add(cui)

# upload on DB
docs = collection.find({})
for mesh in meshid:
    collection.update_many({"MESH": mesh}, {"$set": {"CUI_list": list(mesh_code_to_cui[mesh])}})


# remove empty arrays
docs = collection.find({ 'MESH' : { '$exists': True, '$size': 0 } })
response = []
for doc in docs:
    response.append(doc['_id'])

len(response)

collection.delete_many({'_id': {"$in": response}})

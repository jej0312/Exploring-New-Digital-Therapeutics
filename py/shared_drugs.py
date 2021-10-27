
# ----------------- indications ----------------- #

# import gzip
# import shutil
#
# with gzip.open(r'data/disease/meddra_all_label_indications.tsv.gz', 'rb') as f_in:
#     with open('data/disease/meddra_all_label_indications.tsv', 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)

'''
http://sideeffects.embl.de/download/

CID: drug
C: disease


2: STITCH compound id (flat, see above)
3: UMLS concept id as it was found on the label ** (실제로 사용할 땐 이 칼럼)
4: method of detection: NLP_indication / NLP_precondition / text_mention
5: concept name
6: MedDRA concept type (LLT = lowest level term, PT = preferred term; in a few cases the term is neither LLT nor PT)
7: UMLS concept id for MedDRA term
8: MedDRA concept name
'''


import pickle
# with open('data/disease/psychiatric_disorder_CUIs.pickle', 'rb') as f:
with open('data/disease/mental_and_behavioral_CUIs_final_0726.pickle', 'rb') as f:
    filt = pickle.load(f)

import pandas as pd
dt = pd.read_csv('data/disease/meddra_all_label_indications.tsv', delimiter='\t', header=None)
dt.rename(columns={2: 'drug', 3: 'disease'}, inplace=True)
tmp = dt[['drug', 'disease']]
tmp.drop_duplicates(['drug', 'disease'], inplace=True)
tmp = tmp.loc[tmp['disease'].apply(lambda x: x in filt['cui']), :]

# list 생성
tmp_list = tmp.groupby('drug')['disease'].apply(list)

# imports
import networkx as nx
from operator import itemgetter
from networkx.algorithms.bipartite.matrix import biadjacency_matrix

def to_adjacency_matrix(data):
    g = nx.DiGraph()
    g.add_edges_from(data)
    partition_0 = set(map(itemgetter(0), data))
    partition_1 = set(map(itemgetter(1), data))
    return biadjacency_matrix(g, partition_0).toarray(), partition_0, partition_1

adj_m, drugs, diseases = to_adjacency_matrix(list(tmp.itertuples(index=False)))

import numpy as np
co_drug = np.dot(adj_m.T, adj_m) # disease-disease matrix
pd.DataFrame(co_drug, index = diseases, columns=diseases)

tri_upper_no_diag = np.triu(co_drug, k=1) # diagonal 지우고 upper matrix => 본인-본인 관계 제거, 두 번 반복되는 거 방지
np.argwhere( tri_upper_no_diag > 0 ).shape # shared drug인 disease

# for pair in np.argwhere( tri_upper_no_diag > 0): # disease-disease 쌍 확인
#     print(list(diseases)[pair[0]], "-", list(diseases)[pair[1]], "(", tri_upper_no_diag[pair[0], pair[1]], ")")

for pair in np.argwhere( tri_upper_no_diag == np.max(tri_upper_no_diag) ): # 최대 shared drugs를 가진 pair: 1개
    print(list(diseases)[pair[0]], "-", list(diseases)[pair[1]])

# psychiatric만 추출했을 때
with open("data/disease/shared_drugs_psychiatric_n_behavioral.pickle", "wb") as f:
    pickle.dump(pd.DataFrame(tri_upper_no_diag, index=diseases, columns=diseases), f)

# 전체 다 추출했을 때
with open("data/disease/shared_drugs_all.pickle", "wb") as f:
    pickle.dump(pd.DataFrame(tri_upper_no_diag, index=diseases, columns=diseases), f)


import matplotlib.pyplot as plt
import networkx as nx

edges = [tuple(pair) for pair in np.argwhere( tri_upper_no_diag > 0 )]

graph = nx.DiGraph()
graph.add_nodes_from(diseases)
graph.add_edges_from(edges)

nx.draw(graph, with_labels = True)
plt.show()


# ----------------- indications ----------------- #

# import gzip
# import shutil
#
# with gzip.open(r'data/disease/meddra_all_label_se.tsv.gz', 'rb') as f_in:
#     with open('data/disease/meddra_all_label_se.tsv', 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)

'''
http://sideeffects.embl.de/download/

CID: drug
C: disease

1 & 2: STITCH compound ids (flat/stereo, see above)
3: UMLS concept id as it was found on the label
4: MedDRA concept type (LLT = lowest level term, PT = preferred term; in a few cases the term is neither LLT nor PT)
5: UMLS concept id for MedDRA term
6: side effect name
'''

import pandas as pd
dt = pd.read_csv('data/disease/meddra_all_label_se.tsv', delimiter='\t', header=None)
dt.head(3)
dt.rename(columns={3: 'disease', 4: 'medra_type', 5: 'medra_cui', 6: 'se'}, inplace=True)
# tmp = dt.loc[dt['medra_type'] == 'LLT', ['disease', 'se']]
tmp = dt[['disease', 'se', 'medra_cui']]
tmp.drop_duplicates(['disease', 'se', 'medra_cui'], inplace=True)


import pickle
# with open('data/disease/psychiatric_disorder_CUIs.pickle', 'rb') as f:
#     filt = pickle.load(f)
# tmp = tmp.loc[tmp['disease'].apply(lambda x: x in filt), :]

with open('data/disease/mental_and_behavioral_CUIs_final_0726.pickle', 'rb') as f:
    filt = pickle.load(f)
tmp = tmp.loc[tmp['disease'].apply(lambda x: x in filt['cui']), :]
tmp = tmp.loc[tmp['medra_cui'].apply(lambda x: x in filt['cui']), :]

set(tmp['se'])

tmp.shape

tmp['disease'].nunique()
tmp['se'].nunique()
tmp[tmp['se'].duplicated()]

tmp = tmp[['disease', 'se']]


# list 생성
tmp_list = tmp.groupby('se')['disease'].apply(list)

# imports
import networkx as nx
from operator import itemgetter
from networkx.algorithms.bipartite.matrix import biadjacency_matrix

def to_adjacency_matrix(data):
    g = nx.DiGraph()
    g.add_edges_from(data)
    partition_0 = set(map(itemgetter(0), data))
    partition_1 = set(map(itemgetter(1), data))
    return biadjacency_matrix(g, partition_0).toarray(), partition_0, partition_1

adj_m, diseases, se = to_adjacency_matrix(list(tmp.itertuples(index=False)))



import numpy as np
co_se = np.dot(adj_m, adj_m.T) # disease-disease matrix
co_se.shape

pd.DataFrame(co_se, index=diseases, columns=diseases)

tri_upper_no_diag = np.triu(co_se, k=1) # diagonal 지우고 upper matrix => 본인-본인 관계 제거, 두 번 반복되는 거 방지
np.argwhere( tri_upper_no_diag > 0 ).shape # shared se인 disease

for pair in np.argwhere(tri_upper_no_diag > 0): # disease-disease 쌍 확인
    print(list(diseases)[pair[0]], "-", list(diseases)[pair[1]], "(", tri_upper_no_diag[pair[0], pair[1]], ")")
#
# for pair in np.argwhere( tri_upper_no_diag == np.max(tri_upper_no_diag) ):
#     print(list(diseases)[pair[0]], "-", list(diseases)[pair[1]])


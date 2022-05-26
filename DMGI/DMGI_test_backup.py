# flag = 'all'
flag = 'modeling'
dataset = 'DTx_DMGI'
number_of_groups = 5
weighted = True
nb_epochs = 15000
hid_units = 200
lr = 0.0025
isAttn = False
# drop_prob = 0.5

# %% preprocessing
if (flag == 'preprocessing') | (flag == 'all'):
    import pickle
    import random

    import pandas as pd
    import numpy as np
    import networkx as nx
    import os

    os.getcwd()
    os.chdir(r'C:\Users\SAMSUNG\DMGI')

    """
    data.keys()
    data['feature'].shape # node X feature. in this study, we use hierarchical attribute as label
    data['label'].shape # nodes X label. in this study, we dont use (semi) supervised method, so we use constant 1 for every nodes (3550, 1)
    data['MDM'].shape # node X node. adjacency matrix for each layer
    data['train_idx'].shape # 1 X idx. one array for train idx
    data['val_idx'].shape # 1 X idx. one array for valid idx
    data['test_idx'].shape # 1 X idx. one array for test idx
    """

    # ---- node id ---- #
    import pickle

    with open('data/multiplex/disease_id_DTx_edges_0925.pickle', 'rb') as f:
        idmapper = pickle.load(f)
    len(idmapper.keys())


    def find_id(x):
        return idmapper[x]


    # ---- node feature ---- #
    # get hierarchical structure
    h = pd.read_csv('data/disease/hierarchical_info_0918.csv')
    h['p_cui'] = h.parent.apply(
        lambda x: {k: v for k, v in zip(h.child, h.cui)}[x] if x in {k: v for k, v in zip(h.child, h.cui)} else x)

    leaves = set(h.cui).difference(h.p_cui)
    G = nx.from_pandas_edgelist(h, 'p_cui', 'cui', create_using=nx.DiGraph())
    ancestors = {n: nx.algorithms.dag.ancestors(G, n) for n in leaves}

    feature = (pd.DataFrame.from_dict(ancestors, orient='index')
               .rename(lambda x: 'parent_{}'.format(x + 1), axis=1)
               .rename_axis('child')
               .fillna(''))

    # make feature matrix
    feature_m = pd.DataFrame(index=set(h.cui), columns=set(h.p_cui))
    feature_m.shape
    # feature[(feature == '')] = np.nan
    for i in feature.index:
        feature_m.loc[i, feature.loc[i].values] = 1
    for i in feature_m.columns:
        feature_m.loc[i, i] = 1
    feature_m.fillna(0, inplace=True)
    feature_m.drop(columns=['Mental disorder', ''], inplace=True)
    feature_m.drop(index=['Mental disorder', ''], inplace=True)
    # (feature_m.isna().sum() > 0).sum()

    feature_m.index = feature_m.index.map(find_id)
    feature_m.columns = feature_m.columns.map(find_id)
    feature_m.sort_index(inplace=True)
    feature_m.sort_index(axis=1, inplace=True)
    feature_m = np.array(feature_m)

    # ---- label ---- #
    import numpy

    label = np.array([1] * feature_m.shape[0]).reshape((-1, 1))
    label.shape

    # ---- layers ---- #
    total_df = pd.DataFrame(index=set(h.cui), columns=set(h.cui))
    total_df.fillna(0, inplace=True)
    # (total_df.columns == 'Mental disorder').sum()

    with open("data/disease/shared_technologies_psychiatric.pickle", "rb") as f:
        A1 = pickle.load(f)
    with open("data/disease/shared_drugs_psychiatric.pickle", "rb") as f:
        A2 = pickle.load(f)
    A3 = pd.read_csv("data/disease/gene_similarity_top30.csv")
    with open('data/disease/psychiatric_disorder_CUIs.pickle', 'rb') as f:
        filt = pickle.load(f)

    # chemical, technological layer - unweighted
    if not weighted:
        A1 = (A1 >= 1).astype(int)
    A1 = total_df.combine(A1, np.maximum, fill_value=0, overwrite=False)
    A1 = A1.drop(index=(set(A1.index) - set(total_df.index))).drop(columns=(set(A1.columns) - set(total_df.columns)))
    A1.shape
    A1.index = A1.index.map(find_id)
    A1.columns = A1.columns.map(find_id)
    A1.sort_index(inplace=True)
    A1.sort_index(axis=1, inplace=True)
    A1 = np.array(A1)

    if not weighted:
        A2 = (A2 >= 1).astype(int)
    A2 = total_df.combine(A2, np.maximum, fill_value=0, overwrite=False)
    A2.shape
    A2.index = A2.index.map(find_id)
    A2.columns = A2.columns.map(find_id)
    A2.sort_index(inplace=True)
    A2.sort_index(axis=1, inplace=True)
    A2 = np.array(A2)

    # genetic layer
    A3 = A3.loc[A3['diseaseid1'].apply(lambda x: x in filt), :]
    A3 = A3.loc[A3['diseaseid2'].apply(lambda x: x in filt), :]
    if not weighted:
        A3.jaccard_genes = A3.jaccard_genes.apply(lambda x: x >= 0.1).astype(int)

    import networkx as nx

    edgeList = A3.values.tolist()
    G = nx.DiGraph()
    for i in range(len(edgeList)):
        G.add_edge(edgeList[i][0], edgeList[i][1], weight=edgeList[i][2])
    A3 = nx.adjacency_matrix(G).A
    A3 = np.triu(A3, k=1)
    A3 = pd.DataFrame(A3, columns=list(G.nodes), index=list(G.nodes))
    # A3.loc['C0003469', 'C0011570']
    # edgeList
    A3 = total_df.combine(A3, np.maximum, fill_value=0, overwrite=False)
    A3.shape
    A3.index = A3.index.map(find_id)
    A3.columns = A3.columns.map(find_id)

    A3.sort_index(inplace=True)
    A3.sort_index(axis=1, inplace=True)
    A3 = np.array(A3)

    if number_of_groups == 0:
        # ----------------- train valid test ----------------- #
        import random
        train_idx = list(h.cui.index)
        random.Random(0).shuffle(train_idx)
        test_idx = train_idx[:round(len(train_idx) * 0.2)]
        train_idx = train_idx[round(len(train_idx) * 0.2):]
        val_idx = train_idx[:round(len(train_idx) * 0.3)]
        train_idx = train_idx[round(len(train_idx) * 0.3):]
        print({'train: %d, valid: %d, test: %d' % (len(train_idx), len(val_idx), len(test_idx))})
        train_idx = np.array(train_idx).reshape((1,-1))
        val_idx = np.array(val_idx).reshape((1,-1))
        test_idx = np.array(test_idx).reshape((1,-1))

        # ---- to one data ---- #
        import pickle
        data = {'label': label, 'feature': feature_m,
                'technological': A1, 'chemical': A2, 'genetic': A3,
                'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}

        with open('DMGI/data/{}_{}.pkl'.format(dataset, weighted), 'wb') as f:
            pickle.dump(data, f)

    else:
        # ----------------- k-fold cv ----------------- #
        import pickle
        import random

        train_idx = list(h.cui.index)
        random.Random(0).shuffle(train_idx)
        test_idx = train_idx[:round(len(train_idx) * 0.2)]
        train_idx = train_idx[round(len(train_idx) * 0.2):]


        def divide_data(train_idx, number_of_groups):
            import random
            local_division = len(train_idx) / float(number_of_groups)
            random.Random(0).shuffle(train_idx)
            return [train_idx[int(round(local_division * i)): int(round(local_division * (i + 1)))] for i in
                    range(number_of_groups)]

        separated_data = divide_data(train_idx, number_of_groups)

        for i in range(number_of_groups):
            val_idx = separated_data[i]
            tr_idx = list(set(train_idx) - set(val_idx))
            # train_idx = train_idx[round(len(train_idx) * 0.3):]
            print({'%d split >> train: %d, valid: %d, test: %d' % (i + 1, len(tr_idx), len(val_idx), len(test_idx))})
            tr_idx = np.array(tr_idx).reshape((1, -1))
            val_idx = np.array(val_idx).reshape((1, -1))
            test_idx = np.array(test_idx).reshape((1, -1))

            # ---- to one data ---- #
            data = {'label': label, 'feature': feature_m,
                    'technological': A1, 'chemical': A2, 'genetic': A3,
                    'train_idx': tr_idx, 'val_idx': val_idx, 'test_idx': test_idx}

            with open('DMGI/data/{}_{}_{}_{}.pkl'.format(dataset, weighted, i + 1, number_of_groups), 'wb') as f:
                pickle.dump(data, f)

# %% train embedding
if (flag == 'embedding') | (flag == 'all') | (flag == 'modeling'):
    import numpy as np
    np.random.seed(0)
    import torch
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import argparse
    import sys
    # sys.path.insert(1, r'C:\Users\jej_0\OneDrive - 연세대학교 (Yonsei University)\ISL\01Project\Link Prediction\DMGI')

    def parse_args(k=0):
        # input arguments

        parser = argparse.ArgumentParser(description='DMGI')

        parser.add_argument('--embedder', nargs='?', default='DMGI')
        if k == 0:
            parser.add_argument('--dataset', nargs='?', default='{}_{}'.format(dataset, weighted))
        else:
            parser.add_argument('--dataset', nargs='?', default='{}_{}_{}_{}'.format(dataset, weighted, k, number_of_groups))

        parser.add_argument('--metapaths', nargs='?', default='genetic,chemical,technological')

        parser.add_argument('--nb_epochs', type=int, default=nb_epochs) # default 10000
        parser.add_argument('--hid_units', type=int, default=hid_units) # default 64
        parser.add_argument('--lr', type = float, default = lr) # default 0.0005
        parser.add_argument('--l2_coef', type=float, default=0.0001)
        parser.add_argument('--drop_prob', type=float, default=0.5)
        parser.add_argument('--reg_coef', type=float, default=0.001)
        parser.add_argument('--sup_coef', type=float, default=0.1)
        parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
        parser.add_argument('--margin', type=float, default=0.1)
        parser.add_argument('--gpu_num', type=int, default=0)
        parser.add_argument('--patience', type=int, default=20)
        parser.add_argument('--nheads', type=int, default=1) # attention multi head 수
        parser.add_argument('--activation', nargs='?', default='relu')
        parser.add_argument('--isSemi', action='store_true', default=False)
        parser.add_argument('--isBias', action='store_true', default=False)
        parser.add_argument('--isAttn', action='store_true', default=isAttn)

        return parser.parse_known_args()

    def printConfig(args):
        args_names = []
        args_vals = []
        for arg in vars(args):
            args_names.append(arg)
            args_vals.append(getattr(args, arg))
        print(args_names)
        print(args_vals)

    def main(k=0):
        if k==0:
            args, unknown = parse_args()

            if args.embedder == 'DMGI':
                from models import DMGI
                embedder = DMGI(args)
            elif args.embedder == 'DGI':
                from models import DGI
                embedder = DGI(args)

            embedder.training()

        else:
            for i in range(k):
                args, unknown = parse_args(i+1)

                if args.embedder == 'DMGI':
                    from models import DMGI
                    embedder = DMGI(args)
                elif args.embedder == 'DGI':
                    from models import DGI
                    embedder = DGI(args)

                embedder.training()

    if __name__ == '__main__':
        main(number_of_groups)

# %% link prediction
if (flag == 'link prediction') | (flag == 'all') | (flag == 'modeling'):

    import os.path
    import sys
    sys.path.insert(1, r'C:\Users\SAMSUNG\DMGI\MNE')

    from sklearn.metrics import roc_auc_score
    import math
    import subprocess
    import Node2Vec_LayerSelect
    import numpy as np
    np.random.seed(0)

    from MNE_ import *
    import Random_walk
    import pickle
    from datetime import datetime
    import random

    import torch
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def get_dict_neighbourhood_score(local_model, node1, node2):
        try:
            vector1 = local_model[node1]
            vector2 = local_model[node2]
            return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))  # 코사인 유사도
        except:
            return 2 + random.random()

    def get_dict_AUC(model, true_edges, false_edges):
        true_list = list()
        prediction_list = list()
        for edge in true_edges:
            tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
            true_list.append(1)
            # prediction_list.append(tmp_score)
            # for the unseen pair, we randomly give a prediction
            if tmp_score > 2:
                if tmp_score > 2.5:
                    prediction_list.append(1)
                else:
                    prediction_list.append(-1)
            else:
                prediction_list.append(tmp_score)
        for edge in false_edges:
            tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
            true_list.append(0)
            # prediction_list.append(tmp_score)
            # for the unseen pair, we randomly give a prediction
            if tmp_score > 2:
                if tmp_score > 2.5:
                    prediction_list.append(1)
                else:
                    prediction_list.append(-1)
            else:
                prediction_list.append(tmp_score)
        y_true = np.array(true_list)
        y_scores = np.array(prediction_list)
        return roc_auc_score(y_true, y_scores)

    # randomly divide data into few parts for the purpose of cross-validation
    def divide_data(input_list, group_number):
        local_division = len(input_list) / float(group_number)
        random.Random(0).shuffle(input_list)
        return [input_list[int(round(local_division * i)): int(round(local_division * (i + 1)))] for i in
                range(group_number)]

    # 전체 node 간 조합에서 true edge가 아닌 것을 false edge로 지정
    def randomly_choose_false_edges(nodes, true_edges):
        tmp_list = list()
        all_edges = list()
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                all_edges.append((i, j))
        random.Random(0).shuffle(all_edges)
        for edge in all_edges:
            if edge[0] == edge[1]:
                continue
            if (nodes[edge[0]], nodes[edge[1]]) not in true_edges and (
            nodes[edge[1]], nodes[edge[0]]) not in true_edges:
                tmp_list.append((nodes[edge[0]], nodes[edge[1]]))
        return tmp_list

    def get_AUC_from_edges(edges):
        with open('data/{}_{}_{}_{}.pkl'.format(dataset, weighted, i + 1, number_of_groups), 'rb') as f:
            edges = pickle.load(f)

        test_idx = edges['test_idx'][0]
        true_edges = list(set([tuple(set(edge)) for edge in np.column_stack((test_idx[np.where(
            edges['genetic'][test_idx, :][:, test_idx] > 0)[0]], test_idx[np.where(
            edges['genetic'][test_idx, :][:, test_idx] > 0)[1]])).tolist()] \
                              + [tuple(set(edge)) for edge in np.column_stack((test_idx[np.where(
            edges['chemical'][test_idx, :][:, test_idx] > 0)[0]], test_idx[np.where(
            edges['chemical'][test_idx, :][:, test_idx] > 0)[1]])).tolist()] \
                              + [tuple(set(edge)) for edge in np.column_stack((test_idx[np.where(
            edges['technological'][test_idx, :][:, test_idx] > 0)[0]], test_idx[np.where(
            edges['technological'][test_idx, :][:, test_idx] > 0)[1]])).tolist()]
                              ))

        print('number of true edges:', len(true_edges))
        selected_false_edges = randomly_choose_false_edges(test_idx, true_edges)
        print('number of false edges:', len(selected_false_edges))

        tmp_DMGI_score = get_dict_AUC(dmgi_model, true_edges, selected_false_edges)

        return tmp_DMGI_score

    overall_DMGI_performance = list()

    # train-test-valid
    if number_of_groups == 0:
        with open('revision/best_{}_{}_{}_{}_{}_{}_model.pkl'.format(dataset, weighted, nb_epochs, hid_units, lr, isAttn), 'rb') as f:
            model = pickle.load(f)
        model.load_state_dict(torch.load('DMGI/revision/best_{}_{}_{}_{}_{}_{}.pkl'.format(dataset, weighted, nb_epochs, hid_units, lr, isAttn)))
        dmgi_model = {str(i): model.H.detach().reshape((model.H.detach().shape[1], -1))[i] for i in
                      range(model.H.detach().shape[1])}

        with open('data/{}_{}.pkl'.format(dataset, weighted), 'rb') as f:
            edges = pickle.load(f)

        overall_DMGI_performance = get_AUC_from_edges(edges)

    # k-fold
    else:
        tmp_DMGI_performance = 0

        for i in range(number_of_groups):
            print('Working on {}_{}_{}_{}_{}.pkl'.format(dataset, weighted, i + 1, number_of_groups, isAttn))

            with open(
                    'revision/best_{}_{}_{}_{}_{}_{}_{}_{}_model.pkl'.format(dataset, weighted, i + 1, number_of_groups, nb_epochs, hid_units, lr, isAttn),
                    'rb') as f:
                model = pickle.load(f)
            model.load_state_dict(torch.load(
                'revision/best_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(dataset, weighted, i + 1, number_of_groups, nb_epochs, hid_units, lr, isAttn)))
            dmgi_model = {str(j): model.H.detach().reshape((model.H.detach().shape[1], -1))[j] for j in
                          range(model.H.detach().shape[1])}

            with open('data/{}_{}_{}_{}.pkl'.format(dataset, weighted, i + 1, number_of_groups), 'rb') as f:
                edges = pickle.load(f)

            tmp_DMGI_score = get_AUC_from_edges(edges)
            print('DMGI score:', tmp_DMGI_score)

            overall_DMGI_performance.append(tmp_DMGI_score)

    overall_DMGI_performance = np.asarray(overall_DMGI_performance)
    print('Overall DMGI AUC:', overall_DMGI_performance)

    print('')
    print('')

    print('Overall DMGI AUC:', np.mean(overall_DMGI_performance))
    print('Overall DMGI std:', np.std(overall_DMGI_performance))

    print('end')


# def check_symmetric(a, rtol=1e-05, atol=1e-08):
#     import numpy as np
#     return np.allclose(a, a.T, rtol=rtol, atol=atol)
# check_symmetric(edges['genetic'])



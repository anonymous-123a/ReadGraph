from anomaly_generation import *
from scipy import sparse
import pickle
import time
import os
import argparse
import numpy as np
import shutil

def generateDataset(dataset, snap_size, args, train_per, anomaly_per, on_test=False):
    data_index = args.data_index

    print('Generating data with anomaly for Dataset: ', dataset)

    if args.dataset == 'yelp':
        data_all = np.loadtxt(
                'data/interim/' + f'{dataset}/' +
                dataset + '_' + str(data_index) + '_1' + '.csv',
                dtype=int,
                comments='%',
                delimiter=' ')
    else:
        data_all = np.loadtxt(
            'data/interim/' + f'{dataset}/' +
            dataset + '_' + str(data_index) + '.csv',
            dtype=int,
            comments='%',
            delimiter=' ')

    edges = data_all[:, 0:2].astype(dtype=int)
    vertices = np.unique(edges)
    m = len(edges)
    n = len(vertices)
    idx = list(range(n))
    id_index_map = {vertices[i]: i for i in idx}  # dictionary
    for i in range(m):
        data_all[i, 0] = id_index_map[data_all[i, 0]]
        data_all[i, 1] = id_index_map[data_all[i, 1]]

    id_type_map = {}
    for edge in data_all[:, :3]:
        if edge[0] not in id_type_map:
            id_type_map[edge[0]] = 0  # user
        if edge[1] not in id_type_map:
            if edge[2] == 0:
                id_type_map[edge[1]] = 1  # item
            else:
                id_type_map[edge[1]] = 0

    t0 = time.time()
    if dataset == 'digg_all':
        idx_data_vote = np.nonzero(data_all[:, 2] == 0)
        idx_data_trust = np.nonzero(data_all[:, 2] == 1)
        data_vote = data_all[idx_data_vote]
        data_trust = data_all[idx_data_trust]
        min_item_number = np.min(data_vote[:, 1])
        data_vote[:, 1] = data_vote[:, 1] - min_item_number

        synthetic_train_v, synthetic_test_v = anomaly_generation3(0, anomaly_per,data=data_vote,m=np.size(data_vote,0),
                                                                  max_idx1=np.max(data_vote[:, 0]), max_idx2=np.max(data_vote[:, 1]))
        synthetic_train_u, synthetic_test_u = anomaly_generation3(0, anomaly_per,data=data_trust,m=np.size(data_trust,0),
                                                                  max_idx1=np.max(data_trust[:, 0]), max_idx2=np.max(data_trust[:, 1]))
        synthetic_train_v[:, 4] = 0
        synthetic_train_v[:, 1] = synthetic_train_v[:, 1] + min_item_number
        synthetic_test_v[:, 1] = synthetic_test_v[:, 1] + min_item_number
        synthetic_test_v[:, 4] = 0      #rating 将vote的rating全置为0
        synthetic_test_u[:, 2] = 1
        synthetic_train_u[:, 2] = 1     #trust
        synthetic_data = np.concatenate((synthetic_train_u, synthetic_train_v, synthetic_test_u, synthetic_test_v))
        synthetic_data = synthetic_data[synthetic_data[:, 3].argsort()]
        train_num = int(np.floor(train_per * np.size(synthetic_data, 0)))
        syn_train = synthetic_data[0:train_num, :]
        syn_test = synthetic_data[train_num:, :]
        if args.ano4test:
            nor_test = syn_test[syn_test[:, 5] == 1]
            test_user = nor_test[nor_test[:, 2] == 1]
            test_user = test_user[:, :5]
            _, test_user = anomaly_generation3(0, args.anomaly_per_test, data=test_user,
                                                                  m=np.size(test_user, 0),
                                                                  max_idx1=np.max(test_user[:, 0]),
                                                                  max_idx2=np.max(test_user[:, 1]))
            test_user[:, 4] = 0
            test_user[:, 2] = 1

            test_item = nor_test[nor_test[:, 2] == 0]
            if test_item.size > 0:
                min_item_number_test = np.min(test_item[:, 1])
                test_item[:, 1] = test_item[:, 1] - min_item_number_test
                test_item = test_item[:, :5]
                _, test_item = anomaly_generation3(0, args.anomaly_per_test, data=test_item,
                                                                      m=np.size(test_item, 0),
                                                                      max_idx1=np.max(test_item[:, 0]),
                                                                      max_idx2=np.max(test_item[:, 1]))
                test_item[:, 1] = test_item[:, 1] + min_item_number_test
                test_item[:, 4] = 0
                test_edges = np.concatenate((test_item, test_user))
            else:
                test_edges = test_user
            syn_test = test_edges[np.argsort(test_edges[:, 3])]
            syn_data = np.concatenate((syn_train, syn_test))
        else:
            syn_data = synthetic_data

    elif args.dataset == 'amazon':
        idx_data_vote = np.nonzero(data_all[:, 2] == 0)
        idx_data_trust = np.nonzero(data_all[:, 2] == 1)
        data_vote = data_all[idx_data_vote]
        data_trust = data_all[idx_data_trust]
        min_item_number = np.min(data_vote[:, 1])
        data_vote[:, 1] = data_vote[:, 1] - min_item_number


        synthetic_train_v, synthetic_test_v = anomaly_generation3(0, anomaly_per, data=data_vote,
                                                                  m=np.size(data_vote, 0),
                                                                  max_idx1=np.max(data_vote[:, 0]),
                                                                  max_idx2=np.max(data_vote[:, 1]),
                                                                  datasetname='amazon')

        synthetic_train_v[:, 4] = 0
        synthetic_train_v[:, 1] = synthetic_train_v[:, 1] + min_item_number
        synthetic_test_v[:, 1] = synthetic_test_v[:, 1] + min_item_number
        synthetic_test_v[:, 4] = 0  # rating 将vote的rating全置为0

        synthetic_data = np.concatenate((synthetic_train_v, synthetic_test_v))
        synthetic_data = synthetic_data[synthetic_data[:, 3].argsort()]
        train_num = int(np.floor(train_per * np.size(synthetic_data, 0)))
        syn_train = synthetic_data[0:train_num, :]
        syn_test = synthetic_data[train_num:, :]
        if args.ano4test:
            temp_test = syn_test[syn_test[:, 5] == 1]
            temp_test = temp_test[:, :5]
            min_item_number_test = np.min(temp_test[:, 1])
            temp_test[:, 1] = temp_test[:, 1] - min_item_number_test
            _, syn_test = anomaly_generation3(0, args.anomaly_per_test, data=temp_test,
                                                                  m=np.size(temp_test, 0),
                                                                  max_idx1=np.max(temp_test[:, 0]),
                                                                  max_idx2=np.max(temp_test[:, 1]),
                                                                  datasetname='amazon')
            syn_test[:, 4] = 0
            syn_test[:, 1] = syn_test[:, 1] + min_item_number_test
            syn_test = syn_test[np.argsort(syn_test[:, 3])]
            syn_data = np.concatenate((syn_train, syn_test))

        else:
            syn_data = synthetic_data
    elif args.dataset == 'yelp':
        idx_data_vote = np.nonzero(data_all[:, 2] == 0)
        idx_data_trust = np.nonzero(data_all[:, 2] == 1)
        data_vote = data_all[idx_data_vote]
        data_trust = data_all[idx_data_trust]
        min_item_number = np.min(data_vote[:, 1])

        data_vote[:, 5] = np.where(data_vote[:, 5] == 1, 1, 0)
        synthetic_data = data_vote[data_vote[:, 3].argsort()]
        train_num = int(np.floor(train_per * np.size(synthetic_data, 0)))
        syn_train = synthetic_data[0:train_num, :]
        syn_test = synthetic_data[train_num:, :]
        if args.ano4test :
            syn_test = anomaly_generation_yelp(args.anomaly_per_test, syn_test)
            syn_data = np.concatenate((syn_train, syn_test))
        else:
            syn_data = synthetic_data

    if not os.path.exists('data/percent/' + f'{dataset}'):
        os.makedirs('data/percent/' + f'{dataset}')
    if args.ano4test:
        savedata_path_train = 'data/percent/' + f'{dataset}/' + dataset + '_' + str(data_index) + '_' + str(
            train_per) + '_' + str(anomaly_per) + str(args.anomaly_per_test)+ '_train' + '.csv'
        savedata_path_test = 'data/percent/' + f'{dataset}/' + dataset + '_' + str(data_index) + '_' + str(
            train_per) + '_' + str(anomaly_per) + str(args.anomaly_per_test)+ '_test' + '.csv'

        np.savetxt(savedata_path_train, syn_train,  fmt='%d', delimiter=',')
        np.savetxt(savedata_path_test, syn_test, fmt='%d', delimiter=',')

    savedata_path = 'data/percent/' + f'{dataset}/' + dataset + '_' + str(data_index)+ '_' + str(train_per) + '_' + str(anomaly_per) +'.csv'


    if os.path.exists(savedata_path):
        os.remove(savedata_path)
    np.savetxt(savedata_path, syn_data, fmt='%d', delimiter=',')



    train_mat = csr_matrix((np.ones([np.size(syn_train, 0)], dtype=np.int32), (syn_train[:, 0], syn_train[:, 1])),
                           shape=(n, n))
    train_mat = train_mat + train_mat.transpose()   #【注意】这里把图又看成最普通的图了

    print("Anomaly Generation finish! Time: %.2f s"%(time.time()-t0))
    t0 = time.time()


    train_mat = (train_mat + train_mat.transpose() + sparse.eye(n)).tolil()
    headtail = train_mat.rows
    del train_mat



    train_size = int(len(syn_train) / snap_size + 0.5)
    test_size = int(len(syn_test) / snap_size + 0.5)
    print("Train size:%d  %d  Test size:%d %d" %
          (len(syn_train), train_size, len(syn_test), test_size))
    rows = []
    cols = []
    typs = []
    weis = []
    labs = []
    for ii in range(train_size):
        start_loc = ii * snap_size
        end_loc = (ii + 1) * snap_size

        row = np.array(syn_train[start_loc:end_loc, 0], dtype=np.int32)
        col = np.array(syn_train[start_loc:end_loc, 1], dtype=np.int32)
        typ = np.array(syn_train[start_loc:end_loc, 2], dtype=np.int32)
        lab = np.array(syn_train[start_loc:end_loc, -1], dtype=np.int32)
        wei = np.ones_like(row, dtype=np.int32)

        rows.append(row)
        cols.append(col)
        typs.append((typ))
        weis.append(wei)
        labs.append(lab)

    print("Training dataset contruction finish! Time: %.2f s" % (time.time()-t0))
    t0 = time.time()

    for i in range(test_size):
        start_loc = i * snap_size
        end_loc = (i + 1) * snap_size

        row = np.array(syn_test[start_loc:end_loc, 0], dtype=np.int32)
        col = np.array(syn_test[start_loc:end_loc, 1], dtype=np.int32)
        typ = np.array(syn_test[start_loc:end_loc, 2], dtype=np.int32)
        lab = np.array(syn_test[start_loc:end_loc, -1], dtype=np.int32)
        wei = np.ones_like(row, dtype=np.int32)
        rows.append(row)
        cols.append(col)
        typs.append(typ)
        weis.append(wei)
        labs.append(lab)

    print("Test dataset finish constructing! Time: %.2f s" % (time.time()-t0))
    path_ = os.path.join('data/percent', f'{dataset}')
    if not os.path.exists(path_):
        os.makedirs(path_)
    with open('data/percent/' + f'{dataset}/' + dataset + '_' + str(data_index)+ '_' + str(train_per) + '_' + str(anomaly_per) +'.pkl', 'wb') as f:
        pickle.dump((rows,cols,typs,labs,weis,headtail,id_type_map,train_size,test_size,n,m),f,pickle.HIGHEST_PROTOCOL)
    eigen_file = 'data/eigen/' + f'{args.dataset}/'+ args.dataset + '_' + str(args.data_index) + '_' + str(args.train_ratio) + '_' + str(args.anomaly_ratio) + '.pkl'
    if os.path.exists(eigen_file):
        os.remove(eigen_file)
    GE_path = f'data/temp/{args.dataset}/'
    if os.path.exists(GE_path):
        shutil.rmtree(GE_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['digg_all', 'btc_otc'], default='digg_all')
    parser.add_argument('--anomaly_ratio' ,choices=[0.01, 0.05, 0.1], type=float, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--data_index', type=int, default=0)
    args = parser.parse_args()
    np.random.seed(2020)

    snap_size_dict = {'uci':1000, 'digg_all':200, 'btc_alpha':1000, 'btc_otc':2000}

    if args.anomaly_ratio is None:
        anomaly_pers = [0.01, 0.05, 0.10]
    else:
        anomaly_pers = [args.anomaly_per]

    generateDataset(args.dataset, snap_size_dict[args.dataset], args, train_per=args.train_ratio, anomaly_per=0.10, on_test=True)
    print("prepare_data has done")
import datetime
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix
from sklearn.cluster import SpectralClustering





def edgeList2Adj(data):
    """
    converting edge list to graph adjacency matrix
    :param data: edge list
    :return: adjacency matrix which is symmetric
    """

    data = tuple(map(tuple, data))

    n = int(np.max(data))+1  # Get size of matrix
    matrix = np.zeros((n, n))
    for user, item in data:
        matrix[int(user)][int(item)] = 1  # Convert to 0-based index.【注意】这里是将图看成无向图了，
        matrix[int(item)][int(user)] = 1  # Convert to 0-based index.
    return matrix

def anomaly_generation3(ini_graph_percent, anomaly_percent, data, m, max_idx1, max_idx2, datasetname=None):
    """ generate anomaly
    split the whole graph into training network which includes parts of the
    whole graph edges(with ini_graph_percent) and testing edges that includes
    a ratio of manually injected anomaly edges, here anomaly edges mean that
    they are not shown in previous graph;
     input: ini_graph_percent: percentage of edges in the whole graph will be
                                sampled in the intitial graph for embedding
                                learning
            anomaly_percent: percentage of edges in testing edges pool to be
                              manually injected anomaly edges(previous not
                              shown in the whole graph)
            data: whole graph matrix in sparse form, each row (nodeID,
                  nodeID) is one edge of the graph
            n:  number of total nodes of the whole graph
            m:  number of edges in the whole graph
     output: synthetic_test: the testing edges with injected abnormal edges,
                             each row is one edge (nodeID, nodeID, label),
                             label==0 means the edge is normal one, label ==1
                             means the edge is abnormal;
             train:  the sparse format of the training network, each row
                        (nodeID, nodeID)
    """
    # np.random.seed(1)
    print('Generating anomalous dataset...\n')
    print('Initial network edge percent: ' + str(ini_graph_percent * 100))
    print('\n')
    print('Initial anomaly percent : ' + str(anomaly_percent * 100))
    print('\n')
    train_num = int(np.floor(ini_graph_percent * m))

    # region train and test edges
    # select top train_num edges(0:train_num) as in the training set
    train = data[:train_num, :]
    # train_ = np.unique(train)
    # n_train = len(train_)
    # n = max_idx1 + max_idx2 +2
    adj = np.zeros((max_idx1 + 1, max_idx2+1))
    for edge in train:
        adj[edge[0]][edge[1]] = adj[edge[0]][edge[1]] + 1
        # adj[edge[1]][edge[0]] = adj[edge[1]][edge[0]] + 1
    # nodes=np.unique(data)

    test = data[train_num:, :]

    anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
    idx_test = np.ones([np.size(test, 0) + anomaly_num, 1], dtype=np.int64)
    idx_train = np.ones([np.size(train, 0), 1], dtype=np.int64)
    if datasetname == 'amazon':
        anomaly_pos = np.random.choice(np.arange(2, np.size(idx_test, 0)), anomaly_num, replace=False)
    else:
        anomaly_pos = np.random.choice(np.arange(1, np.size(idx_test, 0)), anomaly_num, replace=False)
    idx_test[anomaly_pos] = 0




    # region Prepare Synthetic test Edges
    idx_anomalies = np.nonzero(idx_test.squeeze() == 0)
    idx_normal = np.nonzero(idx_test.squeeze() == 1)    #1为正常
    test_aedge = np.zeros([np.size(idx_test, 0), 5], dtype=np.int64)
    test_aedge[idx_normal] = test
    # synthetic_test[idx_anomalies, 0:2] = anomalies
    test_edge = processEdges2(idx_anomalies[0], test_aedge, adj, max_idx1, max_idx2, datasetname)
    synthetic_test = np.concatenate((test_edge, idx_test), axis=1)
    synthetic_train = np.concatenate((train, idx_train), axis=1)




    return  synthetic_train, synthetic_test #,n_train

def processEdges2(idx_anomalies, test_aedge, adj, max_idx1, max_idx2, datasetname):
    """
    remove self-loops and duplicates and order edge
    :param fake_edges: generated edge list
    :param data: orginal edge list
    :return: list of edges
    """
    # idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)
    #
    # tmp = fake_edges[idx_fake]
    # tmp[:, [0, 1]] = tmp[:, [1, 0]]  # 调整前后顺序，使得边信息（node1，node2）中node1<=node2
    # fake_edges[idx_fake] = tmp
    for idx in idx_anomalies:
        print(idx)
        flag = 0
        th_1 = np.max(test_aedge[0:idx, 0]) + 1
        th_1_min = np.min(test_aedge[0:idx, 0])
        th_2 = np.max(test_aedge[0:idx, 1]) + 1
        th_2_min = np.min(test_aedge[0:idx, 1])
        # th_1 = max_idx1
        # th_2 = max_idx2
        idx_1 = np.random.choice(th_1, 1, replace=False)[0]
        idx_2 = np.random.choice(th_2, 1, replace=False)[0]
        while idx_1 == idx_2:
            idx_1 = np.random.choice(th_1, 1, replace=False)[0]
            idx_2 = np.random.choice(th_2, 1, replace=False)[0]
        # print(idx_1)
        # print(adj[idx_1][idx_2])
        while adj[idx_1][idx_2] != 0:
            idx_1 = np.random.choice(th_1, 1, replace=False)[0]
            idx_2 = np.random.choice(th_2, 1, replace=False)[0]
            while idx_1 == idx_2:
                idx_1 = np.random.choice(th_1, 1, replace=False)[0]
                idx_2 = np.random.choice(th_2, 1, replace=False)[0]
        while flag == 0:
            for edge in test_aedge[0:idx, :]:
                if idx_1 == edge[0] and idx_2 == edge[1]:
                    flag = 1
                    break
                else:
                    continue
            if flag == 0:
                test_aedge[idx, 0] = idx_1
                test_aedge[idx, 1] = idx_2
                test_aedge[idx, 4] = np.random.randint(2)
                if datasetname == 'epinions' or datasetname == 'amazon':
                    test_aedge[idx, 3] = test_aedge[idx - 1, 3]
                else:
                    test_aedge[idx, 3] = test_aedge[idx - 1, 3] + 1
                break
            else:
                idx_1 = np.random.choice(th_1, 1, replace=False)[0]
                idx_2 = np.random.choice(th_2, 1, replace=False)[0]
                while idx_1 == idx_2:
                    idx_1 = np.random.choice(th_1, 1, replace=False)[0]
                    idx_2 = np.random.choice(th_2, 1, replace=False)[0]
                flag = 0


    return test_aedge



def anomaly_generation_yelp(anomaly_percent, data):
    ano_edges = data[data[:, 5] == 0]
    nor_edges = data[data[:, 5] == 1]


    if anomaly_percent < 0.06:
        ano_num = int(nor_edges.shape[0] * anomaly_percent)
        selected_indices = np.random.choice(ano_edges.shape[0], ano_num, replace=False)
        selected_ano_edges = ano_edges[selected_indices]
        result_edges = np.concatenate((selected_ano_edges, nor_edges))
        result_edges = result_edges[np.argsort(result_edges[:, 3])]
        return result_edges

    elif anomaly_percent > 0.11:
        nor_num = int(ano_edges.shape[0] / anomaly_percent)
        selected_indices = np.random.choice(nor_edges.shape[0], nor_num, replace=False)
        selected_nor_edges = nor_edges[selected_indices]
        result_edges = np.concatenate((selected_nor_edges, ano_edges))
        result_edges = result_edges[np.argsort(result_edges[:, 3])]
        return result_edges

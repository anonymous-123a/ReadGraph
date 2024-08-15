import argparse
import time

import numpy as np
import networkx as nx
import torch
import dgl
from sklearn import metrics
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import copy



def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2', help='specify cuda devices')#

    parser.add_argument('--dataset', type=str, default='yelp',choices=['digg', 'amazon', 'yelp'], help='name of dataset')#
    parser.add_argument('--train_ratio', type=float, default=0.8, help='')
    parser.add_argument('--anomaly_ratio', type=float, default=0.1, help='')
    parser.add_argument('--snap_size', type=int, default=2000, help='')
    parser.add_argument('--data_index', type=int, default=10, help='')
    parser.add_argument('--seed', type=int, default=2020, help='')
    parser.add_argument('--anomaly_per_test', type=float, default=0.15, help='')
    parser.add_argument('--ano4test', type=bool, default=False, help='')

    parser.add_argument('--epochs', type=int, default=1, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--patience', type=int, default=70, help='patience for earlystop')
    parser.add_argument('--patience_delta', type=int, default=0.02, help='')
    parser.add_argument('--onTest', type=bool, default=False, help='is test')

    parser.add_argument('--window_size', type=int, default=3, help='')
    parser.add_argument('--contrastive_window_size', type=int, default=1, help='')  #

    parser.add_argument('--neighbour_num', type=int, default=9, help='')


    parser.add_argument('--layer_num', type=int, default=2, help='han layers')
    parser.add_argument('--x_dim', type=int, default=256, help='input channels of the model')
    parser.add_argument('--h_dim', type=int, default=256, help='hidden channels of the model')
    parser.add_argument('--z_dim', type=int, default=256, help='output channels of the model')
    parser.add_argument('--embedding_dim', type=int, default=128, help='')  #snap中子图的节点的特征的维数
    parser.add_argument('--num_attention_heads', type=int, default=4, help='')  # 求snap中子图的节点的特征时的多头自注意力的头数
    parser.add_argument('--num_heads_HANLayer', type=int, default=8, help='')  # HANLayer中的多头注意力的头数
    parser.add_argument('--hidden_size_HANLayer', type=int, default=8, help='')
    parser.add_argument('--dropout_HANLayer', type=float, default=0.6, help='')

    parser.add_argument('--hidden_units_rh', type=int, default=64, help='')
    parser.add_argument('--relation_hidden_units_rh', type=int, default=8, help='')
    parser.add_argument('--n_layers_rh', type=int, default=2, help='')
    parser.add_argument('--num_heads_rh', type=int, default=8, help='')
    parser.add_argument('--dropout_rh', type=float, default=0.5, help='')
    parser.add_argument('--residual_rh', type=bool, default=True, help='')

    parser.add_argument('--hidden_dim_GRU', type=int, default=256, help='')

    parser.add_argument('--heads_M', type=int, default=8,help='only for GAT heads')
    parser.add_argument('--cross_MLP_layers_M', type=int, default=1, help='')
    parser.add_argument('--M_dim_M', type=int, default=128, help='')
    parser.add_argument('--inner_dim_M', type=int, default=128, help='')
    parser.add_argument('--dropout_M', type=float, default=0, help='')
    parser.add_argument('--layers_num_M', type=int, default=1, help='')
    parser.add_argument('--percent_selected_M', type=float, default=0.7, help='')
    parser.add_argument('--memory_num', type=int, default=4, help='')


    args, unknown = parser.parse_known_args()
    return args


def evaluate(snap_test, trues, preds):
    aucs = {}
    for snap in range(len(snap_test)):
        auc = metrics.roc_auc_score(trues[snap],preds[snap])
        aucs[snap] = auc

    trues_full = np.hstack(trues)
    preds_full = np.hstack(preds)
    auc_full = metrics.roc_auc_score(trues_full, preds_full)

    return aucs, auc_full


def compute_zero_WL(node_list, link_list):
    WL_dict = {}
    for i in node_list:
        WL_dict[i] = 0
    return WL_dict


def generate_hg_amazon(edges, num_nodes_dic, args):
    hg = dgl.heterograph({
        ('user', 'votes', 'item'): ([], []),
        ('item', 're-votes', 'user'): ([], [])
    })
    hg.add_nodes(num_nodes_dic['user'], ntype='user')
    hg.add_nodes(num_nodes_dic['item'], ntype='item')
    for i, edge in enumerate(edges):
        if [edge[0], edge[1]] not in edges[0:i, :2].tolist():
            if  edge[2] == 0:  # vote
                hg.add_edges(edge[0], edge[1], etype='votes')
                hg.add_edges(edge[1], edge[0], etype='re-votes')
            else:
                print('ERROR:def generate_hg_amazon(edges, num_nodes_dic, args)')
                quit()

        else:
            continue
    hg = hg.to(torch.device(args.device))
    return hg


def generate_hg_digg2(edges, num_nodes_dic, args):

    hg = dgl.heterograph({
        ('user','votes','item'): ([], []),
        ('item', 're-votes', 'user'): ([], []),
        ('user', 'trusts', 'user'): ([], []),
        ('user', 're-trusts', 'user'): ([], [])
    })
    hg.add_nodes(num_nodes_dic['user'], ntype='user')
    hg.add_nodes(num_nodes_dic['item'], ntype='item')
    for i, edge in enumerate(edges):
        if [edge[0], edge[1]] not in edges[0:i,:2].tolist():
            if edge[2] == 1:    #trust
                hg.add_edges(edge[0], edge[1], etype='trusts')
                hg.add_edges(edge[1], edge[0], etype='re-trusts')
            elif edge[2] == 0:  #vote
                hg.add_edges(edge[0], edge[1], etype='votes')
                hg.add_edges(edge[1], edge[0], etype='re-votes')
        else:
            continue
    hg = hg.to(torch.device(args.device))
    return hg


def create_metapath_amazon(hg:dgl.DGLGraph, edge, pathnum=3):
    srcnode = edge[0]
    dstnode = edge[1]
    etype = edge[2]
    if etype == 0:#vote UIU UUU IUI
        metapath_src1, _ = dgl.sampling.random_walk(hg, [srcnode] * pathnum,
                                                 metapath=['votes', 're-votes', 'votes', 're-votes'])   #UIUIU
        metapath_dst1, _ = dgl.sampling.random_walk(hg, [dstnode] * pathnum,
                                                 metapath=['re-votes', 'votes', 're-votes', 'votes'])   #IUIUI
        metapaths = torch.concatenate((metapath_src1, metapath_dst1))
    else:
        print('ERROR:def create_metapath_amazon(hg:dgl.DGLGraph, edge, pathnum=3)')
        quit()

    return metapaths


def create_metapath_digg(hg:dgl.DGLGraph, edge, pathnum=3):
    srcnode = edge[0]
    dstnode = edge[1]
    etype = edge[2]
    if etype == 0:#vote UIU UUU IUI
        metapath_src1, _ = dgl.sampling.random_walk(hg, [srcnode] * pathnum,
                                                 metapath=['votes', 're-votes', 'votes', 're-votes'])   #UIUIU
        metapath_src2, _ = dgl.sampling.random_walk(hg, [srcnode] * pathnum,
                                                 metapath=['trusts', 're-trusts', 'trusts', 're-trusts'])    #UUUUU
        metapath_src3, _ = dgl.sampling.random_walk(hg, [srcnode] * pathnum,
                                                    metapath=['re-trusts', 'trusts', 're-trusts', 'trusts'])  # *
        metapath_dst1, _ = dgl.sampling.random_walk(hg, [dstnode] * pathnum,
                                                 metapath=['re-votes', 'votes', 're-votes', 'votes'])   #IUIUI
        metapaths = torch.concatenate((metapath_src1, metapath_src2, metapath_src3, metapath_dst1))
    elif etype == 1:#trust UUU UIU
        metapath_src1, _ = dgl.sampling.random_walk(hg, [srcnode] * pathnum,
                                                 metapath=['trusts', 're-trusts', 'trusts', 're-trusts'])
        metapath_src2, _ = dgl.sampling.random_walk(hg, [srcnode] * pathnum,
                                                 metapath=['votes', 're-votes', 'votes', 're-votes'])
        metapath_src3, _ = dgl.sampling.random_walk(hg, [srcnode] * pathnum,
                                                    metapath=['re-trusts', 'trusts', 're-trusts', 'trusts'])#*
        metapath_dst1, _ = dgl.sampling.random_walk(hg, [dstnode] * pathnum,
                                                 metapath=['re-trusts', 'trusts', 're-trusts', 'trusts'])
        metapath_dst2, _ = dgl.sampling.random_walk(hg, [dstnode] * pathnum,
                                                 metapath=['votes', 're-votes', 'votes', 're-votes'])
        metapath_dst3, _ = dgl.sampling.random_walk(hg, [dstnode] * pathnum,
                                                    metapath=['trusts', 're-trusts', 'trusts', 're-trusts'])#*
        metapaths = torch.concatenate((metapath_src1, metapath_src2, metapath_src3, metapath_dst1, metapath_dst2, metapath_dst3))

    return metapaths

def choose_metapath_nodes(neighbournodes_indexs, metapaths, neighbour_number):
    nodes_number = neighbour_number + 2
    metapath_nodes = torch.unique(metapaths)
    metapath_nodes = np.array(metapath_nodes.to('cpu'))

    if metapath_nodes[0] == -1:
        metapath_nodes = metapath_nodes[1:]

    if np.size(metapath_nodes, 0) < nodes_number:
        for node in neighbournodes_indexs:
            if node not in metapath_nodes:
                metapath_nodes = np.append(metapath_nodes, node)
            else:
                continue
            if np.size(metapath_nodes, 0) == nodes_number:
                break
    elif np.size(metapath_nodes, 0) > nodes_number:
        metapath_nodes = metapath_nodes[:nodes_number]
    metapath_nodes.sort()
    return metapath_nodes


def neighbour_nodes_sample(srcnode, dstnode, id_type_map, hg):

    user_nodes, item_nodes = choose_node_type(id_type_map, [srcnode,dstnode])
    src_neighbour_node_g_1 = dgl.sampling.sample_neighbors(hg, {'user':user_nodes,'item':item_nodes}, -1)
    neighbour_node = np.array([], dtype=int)
    for etype in hg.etypes:
        nodes, _ = src_neighbour_node_g_1.edges(etype= etype)
        neighbour_node = np.concatenate((neighbour_node, (np.array(nodes.to('cpu')))))
    neighbour_node = np.unique(neighbour_node.flatten())

    user_nodes, item_nodes = choose_node_type(id_type_map, neighbour_node)
    src_neighbour_node_g_2 = dgl.sampling.sample_neighbors(hg, {'user':user_nodes,'item':item_nodes}, -1)
    for etype in hg.etypes:
        nodes, _ = src_neighbour_node_g_2.edges(etype= etype)
        neighbour_node = np.concatenate((neighbour_node, (np.array(nodes.to('cpu')))))
    neighbour_node = np.unique(neighbour_node.flatten())
    return neighbour_node

def compute_batch_hop(node_list, edges_all, num_snap, Ss, id_type_map ,args):

    k = args.neighbour_num
    window_size = args.window_size
    batch_hop_dicts = [None] * (window_size-1)
    s_ranking = [0] + list(range(k+1))

    Gs = []
    hgs = []
    num_nodes_dic = {}
    edges_all_arrays = [np.array(edges)[:, 0:2] for edges in edges_all]
    edges_all1 = np.concatenate(edges_all_arrays)
    num_nodes_dic['user'] = np.max(edges_all1) + 1
    num_nodes_dic['item'] = np.max(edges_all1) + 1

    for snap in range(num_snap):
        G = nx.Graph()
        G.add_nodes_from(node_list)
        edges = edges_all[snap]
        G.add_edges_from(edges[:, 0:2])
        Gs.append(G)

        if args.dataset == 'digg_all':
            hg = generate_hg_digg2(edges, num_nodes_dic, args)
        elif args.dataset == 'amazon':
            hg = generate_hg_amazon(edges, num_nodes_dic, args)
        elif args.dataset == 'yelp':
            hg = generate_hg_amazon(edges, num_nodes_dic, args)
        else:
            print('ERROR:compute_batch_hop')
            quit()

        hgs.append(hg)
    hsg_snaps = [None] * (window_size-1)
    for snap in range(window_size - 1, num_snap):
        print(f'snap:{snap}')
        batch_hop_dict = {}
        edges = edges_all[snap]

        hsg_edges = []
        num = 0
        for edge in edges:
            edge_idx = str(snap) + '_' + str(edge[0]) + '_' + str(edge[1])+ '_'+str(num)
            batch_hop_dict[edge_idx] = []
            hsg_esnaps = []
            num = num + 1
            for lookback in range(window_size):
                hg = hgs[snap - lookback]

                if args.dataset == 'digg_all':
                    metapaths = create_metapath_digg(hg, edge, pathnum=4)
                elif args.dataset == 'amazon':
                    metapaths = create_metapath_amazon(hg, edge, pathnum=4)
                elif args.dataset == 'yelp':
                    metapaths = create_metapath_amazon(hg, edge, pathnum=4)
                else:
                    print('ERROR:create_metapath_amazon')
                    quit()

                neighbournodes_indexs = neighbour_nodes_sample(edge[0], edge[1], id_type_map, hg)
                subnode_indexs = choose_metapath_nodes(neighbournodes_indexs=neighbournodes_indexs, metapaths=metapaths, neighbour_number=k)
                mask = (subnode_indexs != edge[0]) & (subnode_indexs != edge[1])
                subnode_indexs = subnode_indexs[mask]


                s = Ss[snap - lookback][edge[0]] + Ss[snap - lookback][edge[1]]

                s[edge[0]] = -1000 # don't pick myself
                s[edge[1]] = -1000 # don't pick myself
                if len(subnode_indexs) < k:
                    top_k_neighbor_index = s.argsort()[-k:][::-1]
                    ii = 0
                    for i in range(len(subnode_indexs), k):
                        while True:
                            if top_k_neighbor_index[ii] in subnode_indexs:
                                ii = ii + 1
                            else:
                                break
                        subnode_indexs = np.append(subnode_indexs, top_k_neighbor_index[ii])
                        ii = ii + 1

                temp_dic = {subnode_indexs[i] : s[subnode_indexs[i]] for i in range(len(subnode_indexs))}
                subnode_indexs = np.array(sorted(temp_dic, key=temp_dic.get, reverse=True))
                subnode_indexs = subnode_indexs[:k]


                indexs = np.hstack((np.array([edge[0], edge[1]]), subnode_indexs))

                user_indexs, item_indexs = choose_node_type(id_type_map,indexs)

                hsg = dgl.node_subgraph(hg, {'user': user_indexs, 'item': item_indexs})

                for i, neighbor_index in enumerate(indexs):
                    try:
                        hop1 = nx.shortest_path_length(Gs[snap-lookback], source=edge[0], target=neighbor_index)
                    except:
                        hop1 = 99
                    try:
                        hop2 = nx.shortest_path_length(Gs[snap-lookback], source=edge[1], target=neighbor_index)
                    except:
                        hop2 = 99
                    hop = min(hop1, hop2)
                    batch_hop_dict[edge_idx].append((neighbor_index, id_type_map[neighbor_index] ,s_ranking[i], hop, lookback))
                hsg_esnaps.append(hsg)
            hsg_edges.append(hsg_esnaps)
        batch_hop_dicts.append(batch_hop_dict)
        hsg_snaps.append(hsg_edges)
    return batch_hop_dicts, hsg_snaps


def dicts_to_embeddings(feats, batch_hop_dicts, wl_dict, num_snap, args, use_raw_feat=False):

    raw_embeddings = []
    wl_embeddings = []
    hop_embeddings = []
    int_embeddings = []
    time_embeddings = []
    type_embeddings = []
    neighbour_snaps = []
    edges_snaps = []

    for snap in range(num_snap):

        batch_hop_dict = batch_hop_dicts[snap]

        if batch_hop_dict is None:
            raw_embeddings.append(None)
            wl_embeddings.append(None)
            hop_embeddings.append(None)
            int_embeddings.append(None)
            time_embeddings.append(None)
            type_embeddings.append(None)
            neighbour_snaps.append(None)
            edges_snaps.append(None)
            continue

        raw_features_list = []
        role_ids_list = []
        position_ids_list = []
        hop_ids_list = []
        time_ids_list = []
        type_ids_list = []
        neighbour_edges = []
        edges = []


        for edge_idx in batch_hop_dict:

            neighbors_list = batch_hop_dict[edge_idx]
            edge = edge_idx.split('_')[1:3]
            edge[0], edge[1] = int(edge[0]), int(edge[1])

            raw_features = []
            role_ids = []
            position_ids = []
            hop_ids = []
            time_ids = []
            type_ids = []
            neighbour_esnaps = []

            time_num = 1
            raw_feature = []
            role_id = []
            hop_id = []
            position_id = []
            time_id = []
            type_id = []
            neighbour_id = []
            for neighbor,type, intimacy_rank, hop, time in neighbors_list:

                if time_num == time:
                    raw_features.append(raw_feature)
                    role_ids.append(role_id)
                    hop_ids.append(hop_id)
                    position_ids.append(position_id)
                    time_ids.append(time_id)
                    type_ids.append(type_id)
                    neighbour_esnaps.append(neighbour_id)

                    time_num = time_num + 1
                    raw_feature = []
                    role_id = []
                    hop_id = []
                    position_id = []
                    time_id = []
                    type_id = []
                    neighbour_id = []


                if use_raw_feat:
                    raw_feature.append(feats[snap-time][neighbor])
                else:
                    raw_feature.append(None)
                role_id.append(wl_dict[neighbor])
                hop_id.append(hop)
                position_id.append(intimacy_rank)
                time_id.append(time)
                type_id.append(type)
                neighbour_id.append(neighbor)
            role_ids.append(role_id)
            hop_ids.append(hop_id)
            position_ids.append(position_id)
            time_ids.append(time_id)
            type_ids.append(type_id)
            neighbour_esnaps.append(neighbour_id)

            raw_features_list.append(raw_features)
            role_ids_list.append(role_ids)
            position_ids_list.append(position_ids)
            hop_ids_list.append(hop_ids)
            time_ids_list.append(time_ids)
            type_ids_list.append(type_ids)
            neighbour_edges.append(neighbour_esnaps)

            edges.append([edge[0], edge[1]])

        if use_raw_feat:
            raw_embedding = torch.FloatTensor(raw_features_list)
        else:
            raw_embedding = None

        wl_embedding = torch.LongTensor(role_ids_list)
        hop_embedding = torch.LongTensor(hop_ids_list)
        int_embedding = torch.LongTensor(position_ids_list)
        time_embedding = torch.LongTensor(time_ids_list)
        type_embedding = torch.LongTensor(type_ids_list)
        neighbour_edges = torch.LongTensor(neighbour_edges)
        edges = torch.LongTensor(edges)

        raw_embeddings.append(raw_embedding)
        wl_embeddings.append(wl_embedding)
        hop_embeddings.append(hop_embedding)
        int_embeddings.append(int_embedding)
        time_embeddings.append(time_embedding)
        type_embeddings.append(type_embedding)
        neighbour_snaps.append(neighbour_edges)
        edges_snaps.append(edges)

    return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings, type_embeddings, neighbour_snaps, edges_snaps


def generate_embedding(data, args):
    num_snap = len(data['edges'])
    WL_dict = compute_zero_WL(data['idx'], np.vstack(data['edges'][:7]))
    batch_hop_dicts, hsg_snaps = compute_batch_hop(data['idx'], data['edges'], num_snap, data['S'], data['id_type_map'], args)
    raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings, type_embeddings, neighbour_snaps, edges_snaps= \
        dicts_to_embeddings(data['X'], batch_hop_dicts, WL_dict, num_snap, args)
    return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings, type_embeddings, neighbour_snaps, hsg_snaps, edges_snaps


def EdgesToNodes(edges_embeddings, edges_snap, all_z, all_node_idx):
    last_z = all_z[-1]
    nodeID_to_embedding = {}
    nodeID_to_embedding_num = {}
    z = last_z.detach()
    node_index = torch.unique(edges_snap)

    for i, edge in enumerate(edges_snap):
        for ii in range(2):
            node_id = edge[ii].item()
            if node_id in nodeID_to_embedding:
                nodeID_to_embedding[node_id] = nodeID_to_embedding[node_id] + edges_embeddings[i]
                nodeID_to_embedding_num[node_id] = nodeID_to_embedding_num[node_id] + 1
            else:
                nodeID_to_embedding[node_id] = edges_embeddings[i]
                nodeID_to_embedding_num[node_id] = 1

    for i in node_index:
        node_id = i.item()
        if node_id in nodeID_to_embedding:
            z[i] = nodeID_to_embedding[node_id] / nodeID_to_embedding_num[node_id]

    last_z = all_z[-1].detach()
    all_z[-1] = last_z
    all_z.append(z)
    all_node_idx.append(node_index)

    return all_z, all_node_idx


def choose_node_type(id_type_map:dict, nodes):
    user_node = []
    item_node = []
    for node in nodes:
        if id_type_map[node] == 0:    #user
            user_node.append(node)
        elif id_type_map[node] == 1:    #item
            item_node.append(node)
    return user_node, item_node



def get_n_params(model: nn.Module):

    return sum(p.numel() for p in model.parameters())


def calucate_emb(X_in_dim,M_dim,inner_dim,X,M):

    to_q = nn.Linear(M_dim, inner_dim, bias=False).to(M)
    to_k = nn.Linear(X_in_dim, inner_dim, bias=False).to(M)

    q = to_q(M)
    k = to_k(X)

    sim = einsum('i d,j d -> i j', q, k) * (inner_dim ** -0.5)

    sim = F.softmax(sim, dim=0)
    sim = torch.transpose(sim, 0, 1)
    out_ebs = torch.matmul(sim, M)
    out_ebs = nn.Linear(M_dim, X_in_dim, bias=False).to(M)(out_ebs)
    k_expanded = k.unsqueeze(1)
    M_expanded = M.unsqueeze(0)

    com_loss = feature_compactness_loss(k, sim, M)

    result = torch.norm(k_expanded - M_expanded, dim=2, p=2)
    zloss = com_loss
    Result = {}
    Result['loss'] = result
    Result['k'] = k
    Result['M'] = M


    return out_ebs, Result, zloss

def feature_compactness_loss(X, sim, M):
    nearest_mem_idx = torch.argmax(sim, dim=1)
    nearest_mem = M[nearest_mem_idx]

    # Calculate the compactness loss
    compactness_loss = torch.sum(torch.norm(X - nearest_mem, dim=1))

    return compactness_loss

def calucate_sim(X_in_dim,M_dim,inner_dim,X,M):

    to_q = nn.Linear(M_dim, inner_dim, bias=False).to(M)
    to_k = nn.Linear(X_in_dim, inner_dim, bias=False).to(M)

    q = to_q(M)
    k = to_k(X)

    sim = einsum('i d,j d -> i j', q, k) * (inner_dim ** -0.5)


    sim = F.softmax(sim, dim=-1)
    sim = torch.sum(sim, dim=0).squeeze()

    return sim

def process_hsg_edges(hsg_edges, outputs, type_edges):
    for i in range(outputs.size(0)):
        for ii in range(outputs.size(1)):
            embeddings = outputs[i][ii]
            type_nodes = type_edges[i][ii]

            user_embeddings = embeddings[type_nodes == 0]
            item_embeddings = embeddings[type_nodes == 1]

            hsg_edges[i][ii].ndata['feat'] = {'user': user_embeddings.detach(), 'item': item_embeddings.detach()}

    return hsg_edges

def process_combined_hsg(hsg_combined):
    g = hsg_combined

    edges_to_remove = [('user', 'isolated', 'user'), ('item', 'isolated', 'item')]


    new_data_dict = {}
    for etype in g.canonical_etypes:
        if etype not in edges_to_remove:
            src, dst = g.all_edges(form='uv', etype=etype)
            new_data_dict[etype] = (src, dst)


    new_g = dgl.heterograph(new_data_dict, num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes})


    for ntype in g.ntypes:
        if 'feat' in g.nodes[ntype].data:
            new_g.nodes[ntype].data['feat'] = g.nodes[ntype].data['feat'].detach()

    return new_g

def fill_mask(mask, len):

    result_tensor = torch.empty(mask.shape[0], mask.shape[1], len)

    result_tensor.fill_(-1e9)

    ones_tensor = torch.ones(len)

    indices = (mask == 1).nonzero(as_tuple=True)

    result_tensor[indices[0], indices[1], :] = ones_tensor

    return result_tensor

def fill_mask1(mask, len):

    result_tensor = torch.zeros(mask.shape[0], len)

    indices = (mask == 1).nonzero(as_tuple=True)[0]

    result_tensor[indices] = 1

    return result_tensor

def prepare_hsg_snaps(args, hsg_snaps, hop_embeddings, type_embeddings):
    device = args.device
    if args.dataset == 'digg_all':
        graph = dgl.heterograph({
            ('user', 'votes', 'item'): ([], []),
            ('item', 're-votes', 'user'): ([], []),
            ('user', 'trusts', 'user'): ([], []),
            ('user', 're-trusts', 'user'): ([], [])
        })
    elif args.dataset == 'amazon':
        graph = dgl.heterograph({
            ('user', 'votes', 'item'): ([], []),
            ('item', 're-votes', 'user'): ([], [])
        })
    elif args.dataset == 'yelp':
        graph = dgl.heterograph({
            ('user', 'votes', 'item'): ([], []),
            ('item', 're-votes', 'user'): ([], [])
        })
    else:
        print('ERROR:def prepare_hsg_snaps(args, hsg_snaps):')
        quit()
    sample_nodes_num = [{etype: -1 for etype in graph.canonical_etypes} for _ in range(args.n_layers_rh)]
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_nodes_num)

    mask_hsg_rs = []
    for iii, hsg_edges in enumerate(hsg_snaps):
        if hsg_edges is None:
            mask_hsg_rs.append(None)
            continue

        hsg_edges_size0 = len(hsg_edges)
        hsg_edges_size1 = len(hsg_edges[0])
        canonical_etypes = graph.canonical_etypes
        hop_embedding = hop_embeddings[iii]
        type_embedding =type_embeddings[iii]
        gl_embedding = torch.where(hop_embedding != 99, torch.tensor(1), torch.tensor(0))
        with torch.no_grad():

            mask = {etype: np.zeros([hsg_edges_size0, hsg_edges_size1], dtype=int) for etype in canonical_etypes}
            user_offset = {etype: 0 for etype in canonical_etypes}
            item_offset = {etype: 0 for etype in canonical_etypes}

            idx_num_all = {etype: np.zeros([hsg_edges_size0, hsg_edges_size1], dtype=int) for etype in canonical_etypes}

            offsets = {etype: np.zeros([hsg_edges_size0, hsg_edges_size1], dtype=int) for etype in canonical_etypes}
            user_mask = [{etype: [] for etype in canonical_etypes}]
            user_mask = [copy.deepcopy(user_mask[0]) for _ in range(args.n_layers_rh)]
            item_mask = [{etype: [] for etype in canonical_etypes}]
            item_mask = [copy.deepcopy(item_mask[0]) for _ in range(args.n_layers_rh)]
            temp_user_num = [{etype : 0 for etype in canonical_etypes}, {etype: 0 for etype in canonical_etypes}]
            mask4node = {etype: [[0 for _ in range(hsg_edges_size1)] for _ in range(hsg_edges_size0)] for etype in canonical_etypes}
            for i, hsg_snaps in enumerate(hsg_edges):
                for ii, hsg in enumerate(hsg_snaps):
                    user_idx = hsg.nodes('user').to(device)
                    item_idx = hsg.nodes('item').to(device)

                    if user_idx.size(0) > 0:
                        user_blocks = block_sampler(hsg, user_idx, 'user', device, sampler)
                        len_user_blocks = len(user_blocks)

                        for k in range(len_user_blocks):

                            for (stype, etype, dtype) in user_blocks[k].canonical_etypes:
                                etype = (stype, etype, dtype)
                                rel_graph = user_blocks[k][etype]
                                if rel_graph.number_of_edges() != 0:
                                    node_num = int(user_blocks[k].number_of_dst_nodes(dtype))
                                    temp_user_num[k][etype] = temp_user_num[k][etype] + node_num
                                    user_mask[k][etype].extend([1] * node_num)
                                else:
                                    node_num = int(user_blocks[k].number_of_dst_nodes(dtype))
                                    temp_user_num[k][etype] = temp_user_num[k][etype] + node_num
                                    user_mask[k][etype].extend([0] * node_num)

                        for etype in user_blocks[-1].canonical_etypes:
                            rel_graph = user_blocks[-1][etype]
                            if rel_graph.number_of_edges() != 0:
                                mask[etype][i][ii] = 1
                                idx_num_all[etype][i][ii] = user_idx.size(0)
                                offsets[etype][i][ii] = user_offset[etype]
                                user_offset[etype] += user_idx.size(0)
                                temp1 = gl_embedding[i][ii][type_embedding[i][ii] == 0]
                                mask4node[etype][i][ii] = temp1.tolist()
                            else:
                                user_offset[etype] += user_idx.size(0)

                    if item_idx.size(0) > 0:
                        item_blocks = block_sampler(hsg, item_idx, 'item', device, sampler)
                        len_item_blocks = len(item_blocks)

                        for k in range(len_item_blocks):

                            for (stype, etype, dtype) in item_blocks[k].canonical_etypes:
                                etype = (stype, etype, dtype)
                                rel_graph = item_blocks[k][etype]
                                if rel_graph.number_of_edges() != 0:
                                    node_num = int(item_blocks[k].number_of_dst_nodes(dtype))
                                    # node_num = int(rel_graph.num_dst_nodes())
                                    item_mask[k][etype].extend([1] * node_num)
                                else:
                                    node_num = int(item_blocks[k].number_of_dst_nodes(dtype))
                                    # node_num = int(rel_graph.num_dst_nodes())
                                    item_mask[k][etype].extend([0] * node_num)

                        for etype in item_blocks[-1].canonical_etypes:
                            rel_graph = item_blocks[-1][etype]
                            if rel_graph.number_of_edges() != 0:
                                mask[etype][i][ii] = 1
                                idx_num_all[etype][i][ii] = item_idx.size(0)
                                offsets[etype][i][ii] = item_offset[etype]
                                item_offset[etype] += item_idx.size(0)
                                temp1 = gl_embedding[i][ii][type_embedding[i][ii] == 1]
                                mask4node[etype][i][ii] = temp1.tolist()
                            else:
                                item_offset[etype] += item_idx.size(0)

                    torch.cuda.empty_cache()

        mask_hsg_rs.append({'user_mask': user_mask, 'item_mask': item_mask, 'mask':mask,
                            'offsets': offsets, 'idx_num_all': idx_num_all, 'mask4node':mask4node})
    return mask_hsg_rs

def prepare_combined_hsgs(args, hsg_snaps):
    snap_size = args.snap_size
    if args.onTest:
        snap_size = snap_size * 0.1
    window_size = args.window_size

    combined_hsgs = []
    for hsg_edges in hsg_snaps:
        if hsg_edges is None:
            combined_hsgs.append(None)
            continue

        all_hsg_graphs = []
        user_offset = 0
        item_offset = 0
        user_offsets = []
        item_offsets = []
        all_user_ids = []
        all_item_ids = []
        time0 = time.time()

        # Collect all embeddings, type nodes, hsg edges, and node IDs
        for i in range(len(hsg_edges)):
            for ii in range(window_size):
                hsg = hsg_edges[i][ii]

                all_hsg_graphs.append(hsg)

                user_ids = hsg.nodes('user') + user_offset
                item_ids = hsg.nodes('item') + item_offset
                all_user_ids.append(user_ids)
                all_item_ids.append(item_ids)

                user_offsets.append(user_offset)
                item_offsets.append(item_offset)

                user_offset += hsg.num_nodes('user')
                item_offset += hsg.num_nodes('item')
        time1 = time.time()
        print(f'Collect all embeddings, type nodes, hsg edges, and node IDs uses {time1 - time0}')
        # Concatenate all embeddings and type nodes

        # Collect all node IDs
        all_user_ids = torch.cat(all_user_ids)
        all_item_ids = torch.cat(all_item_ids)

        # Create a combined graph
        combined_graph_data = {}
        for hsg, u_offset, i_offset in zip(all_hsg_graphs, user_offsets, item_offsets):
            for srctype, etype, dsttype in hsg.canonical_etypes:
                src, dst = hsg.all_edges(order='eid', etype=etype)
                if srctype == 'user':
                    src += u_offset
                else:
                    src += i_offset
                if dsttype == 'user':
                    dst += u_offset
                else:
                    dst += i_offset
                if (srctype, etype, dsttype) not in combined_graph_data:
                    combined_graph_data[(srctype, etype, dsttype)] = ([], [])
                combined_graph_data[(srctype, etype, dsttype)][0].append(src)
                combined_graph_data[(srctype, etype, dsttype)][1].append(dst)

        for key in combined_graph_data.keys():
            combined_graph_data[key] = (torch.cat(combined_graph_data[key][0]), torch.cat(combined_graph_data[key][1]))
        time7 = time.time()
        print(f'two for uses {time7 - time1}')

        # Include isolated nodes in the graph
        combined_graph_data[('user', 'isolated', 'user')] = (all_user_ids, all_user_ids)  # 上面两个for，显存7060-》7140，看看时间能不能优化
        combined_graph_data[('item', 'isolated', 'item')] = (all_item_ids, all_item_ids)
        combined_hsgs.append(combined_graph_data)

    return combined_hsgs


def block_sampler(hsg, node_idx, ntype, device, sampler):
    loader = dgl.dataloading.DataLoader(hsg, {ntype: node_idx}, sampler, batch_size=node_idx.size(0),
                                        drop_last=False, device=device)

    for i, (input_nodes, output_nodes, blocks) in enumerate(loader):
        blocks = [block.to(device) for block in blocks]

    return blocks

def nowdt():
    """
    get string representation of date and time of now()
    """
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")



import time

import torch
import torch.nn as nn
from Component import *
from torch_geometric.nn import InnerProductDecoder
from utils import *
import torch.nn.functional as F
# from torchviz import make_dot


class Model(nn.Module):
    warm_up = 0
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        config = MyConfig(k=args.neighbour_num, window_size=args.window_size, hidden_size=args.embedding_dim,
                     intermediate_size=args.embedding_dim, num_attention_heads=args.num_attention_heads,
                     weight_decay=args.weight_decay, snaps_num=args.window_size)

        self.node_encoding = NodeEncodeModel(args, config)
        self.graph_encoding2 = GraphEncodeModel_HAN2(args, config)
        self.predict = nn.Linear(args.hidden_size_HANLayer * args.num_heads_HANLayer, 1)
        self.contrastive2 = Contrastive2(args.device, args)
        self.dec = InnerProductDecoder()
        self.relational_subgraph_encoding2 = RelationalSubgraphEncoder2(args=args)
        self.landmark_matching = LandmarkMarkMatchs(args=args)

    def forward(self, y, id_type_map, int_embedding, hop_embedding, time_embedding, type_embedding, neighbours_edges,
                 edges_snap, all_z, all_node_idx, mask_hsg_rs=None, combined_graph_data=None, edges_type=None, istest=False):


        time0 = time.time()
        outputs_edges = self.node_encoding(int_embedding, hop_embedding, time_embedding, type_embedding)
        time1=time.time()
        print(f'model node_encoding time:{time1-time0}')

        time2 = time.time()

        edges_embeddings, hsg_combined = self.graph_encoding2(outputs_edges, type_embedding, combined_graph_data, hop_embedding)

        time3 = time.time()
        print(f'model graph_encoding time:{time3-time2}')

        R_graph_embeddings_edges = self.relational_subgraph_encoding2(combined_hsg= hsg_combined, mask_hsg_rs=mask_hsg_rs, hsg_edges_size0=outputs_edges.size(0))
        time4 = time.time()
        print(f'model relational_subgraph_encoding time:{time4-time3}')

        predict2, sim_cs, zloss = self.landmark_matching(R_graph_embeddings_edges = R_graph_embeddings_edges, edges_embeddings = edges_embeddings, istest=istest)#显存基本没变17812

        predict2 = nn.Sigmoid()(predict2)
        time5 = time.time()
        print(f'model landmark_matching time:{time5 - time4}')

        predict1 = self.predict(edges_embeddings)
        predict1 = nn.Sigmoid()(predict1)
        predict = 0.5 * predict1 + 0.5 * predict2
        bce_loss = F.binary_cross_entropy(predict, y, reduction='none')


        all_z, all_node_idx = EdgesToNodes(edges_embeddings, edges_snap, all_z, all_node_idx)
        label_rectifier = self.dec(all_z[-1], edges_snap.t(), sigmoid=True)
        label_rectifier = label_rectifier.unsqueeze(1)
        reg_loss = torch.norm(label_rectifier - predict1, dim=1, p=2).mean()
        reg_loss = reg_loss + zloss

        time4 = time.time()

        if len(all_z) > self.args.window_size:
            nce_loss = self.contrastive2(all_z, all_node_idx)
        else:
            nce_loss = 0
        time5 = time.time()
        print(f"model contrastive:{time5 - time4}")

        return bce_loss, reg_loss,nce_loss, all_z, all_node_idx, predict
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch import einsum
from einops import rearrange, repeat
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPooler, BertAttention, BertIntermediate, BertOutput
from transformers.configuration_utils import PretrainedConfig
from dgl.nn.pytorch import GATConv
import dgl
from dgl.ops import edge_softmax
import dgl.function as fn
from utils import process_combined_hsg, calucate_sim, fill_mask, fill_mask1, calucate_emb
import numpy as np
import copy

TransformerLayerNorm = torch.nn.LayerNorm

class MyConfig(PretrainedConfig):

    def __init__(
        self,
        k=5,
        max_hop_dis_index = 100,
        max_inti_pos_index = 100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.3,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_decoder=False,
        batch_size = 256,
        window_size = 1,
        weight_decay = 5e-4,
        node_type_num = 2,
        snaps_num = 5,
        **kwargs
    ):
        super(MyConfig, self).__init__(**kwargs)
        self.max_hop_dis_index = max_hop_dis_index
        self.max_inti_pos_index = max_inti_pos_index
        self.k = k
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder
        self.batch_size = batch_size
        self.window_size = window_size
        self.weight_decay = weight_decay
        self.node_type_num = node_type_num
        self.snaps_num = snaps_num


class Contrastive2(nn.Module):
    def __init__(self, device, args):
        super(Contrastive2, self).__init__()
        self.device = device
        self.max_dis = args.contrastive_window_size
        self.start_window = args.window_size - 1
        z_dim = args.hidden_size_HANLayer * args.num_heads_HANLayer
        self.linear = nn.Linear(z_dim, z_dim)

    def forward(self, all_z, all_node_idx):
        t_len = len(all_node_idx)
        nce_loss = 0
        f = lambda x: torch.exp(x)

        s = t_len - 1
        for i in range(self.start_window, s):
            pre_s = i
            nodes_1, nodes_2 = all_node_idx[s].tolist(), all_node_idx[pre_s].tolist()
            common_nodes = list(set(nodes_1) & set(nodes_2))
            if len(common_nodes) == 0:
                continue
            z_anchor = all_z[s][common_nodes]
            z_anchor = self.linear(z_anchor)
            positive_samples = all_z[pre_s][common_nodes]
            pos_sim = f(self.sim(z_anchor, positive_samples, True))
            neg_sim = f(self.sim(z_anchor, all_z[pre_s], False))
            neg_sim = neg_sim.sum(dim=-1).unsqueeze(1)
            nce_loss += -torch.log(pos_sim / (neg_sim)).mean()

        return nce_loss / (s - self.start_window)

    def sim(self, h1, h2, pos=False):
        z1 = F.normalize(h1, dim=-1, p=2)
        z2 = F.normalize(h2, dim=-1, p=2)
        if pos == True:
            return torch.einsum('ik, ik -> i', z1, z2).unsqueeze(1)
        else:
            return torch.mm(z1, z2.t())

class EarlyStopping:
    def __init__(self, args, patience=7, verbose=False, delta=0):
        self.args = args
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = 0
        self.delta = delta

    def __call__(self, val_auc, model):
        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when validation AUC increase.'''
        if self.verbose:
            print(f'Validation AUC increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.args.checkpoints_path)
        self.val_auc_max = val_auc


class NodeEncodeModel(BertPreTrainedModel):
    def __init__(self, args, config):
        super(NodeEncodeModel, self).__init__(config)
        self.config = config
        self.args = args

        self.embeddings = NodeEncoding(config)
        self.encoder = TransformerEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, type_dis_ids, head_mask=None):
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        embeddings_output = self.embeddings(
            init_pos_ids=init_pos_ids,
            hop_dis_ids=hop_dis_ids,
            time_dis_ids=time_dis_ids,
            type_dis_ids=type_dis_ids
        )

        batch_size, num_snapshots, num_nodes, num_types, embedding_dim = embeddings_output.size()

        # Initialize the encoder outputs tensor on the device
        encoder_outputs = torch.empty((batch_size, num_snapshots, num_nodes, embedding_dim), device=self.args.device)

        for i in range(num_snapshots):
            snapshot_embeddings = embeddings_output[:, i, :, :, :].reshape(batch_size * num_nodes, num_types,
                                                                        embedding_dim).to(self.args.device)

            encoder_nodes = self.encoder(snapshot_embeddings, head_mask=head_mask)[
                0]

            pooled_out_nodes = self.pooler(encoder_nodes)

            pooled_out_nodes = pooled_out_nodes.view(batch_size, num_nodes, embedding_dim)

            encoder_outputs[:, i, :, :] = pooled_out_nodes
        return encoder_outputs


class NodeEncoding(nn.Module):
    def __init__(self, config):
        super(NodeEncoding, self).__init__()
        self.config = config

        self.inti_pos_embeddings = nn.Embedding(config.max_inti_pos_index, config.hidden_size)
        self.hop_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)
        self.time_dis_embeddings = nn.Embedding(config.snaps_num, config.hidden_size)
        self.type_dis_embeddings = nn.Embedding(config.node_type_num, config.hidden_size)

        self.LayerNorm = TransformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, init_pos_ids=None, hop_dis_ids=None, time_dis_ids=None, type_dis_ids=None):
        position_embeddings = self.inti_pos_embeddings(init_pos_ids)
        hop_embeddings = self.hop_dis_embeddings(hop_dis_ids)
        time_embeddings = self.time_dis_embeddings(time_dis_ids)
        type_embeddings = self.type_dis_embeddings(type_dis_ids)

        # Layer normalization and dropout
        position_embeddings = self.dropout(self.LayerNorm(position_embeddings))
        hop_embeddings = self.dropout(self.LayerNorm(hop_embeddings))
        time_embeddings = self.dropout(self.LayerNorm(time_embeddings))
        type_embeddings = self.dropout(self.LayerNorm(type_embeddings))

        # Concatenate embeddings along the last dimension
        embeddings = torch.cat((position_embeddings, hop_embeddings, time_embeddings, type_embeddings), dim=-1).contiguous()


        # Reshape the concatenated embeddings
        embeddings = embeddings.view(
            position_embeddings.size(0), position_embeddings.size(1), position_embeddings.size(2), 4, position_embeddings.size(3)#
        )

        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = []
        all_attentions = []

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i] if head_mask is not None else None,
                encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions.append(layer_outputs[1])

        if self.output_hidden_states:
            all_hidden_states.append(hidden_states)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs += (all_hidden_states,)
        if self.output_attentions:
            outputs += (all_attentions,)

        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs += cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)  # (N, M, 1)
        beta = torch.softmax(w.squeeze(-1), dim=1)  # (N, M)
        return (beta.unsqueeze(-1) * z).sum(dim=1)  # (N, D)


class HANLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))

        # Ensure the tensors are contiguous before stacking
        semantic_embeddings = [emb.contiguous() for emb in semantic_embeddings]
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)

class GraphEncodeModel_HAN2(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        if args.dataset == 'digg_all':
            self.HANLayer_user = HANLayer([['votes', 're-votes'], ['trusts', 're-trusts'], ['re-trusts', 'trusts']],
                                          in_size=args.embedding_dim, out_size=args.hidden_size_HANLayer,
                                          layer_num_heads=args.num_heads_HANLayer, dropout=args.dropout_HANLayer)
        elif args.dataset == 'amazon':
            self.HANLayer_user = HANLayer([['votes', 're-votes']],
                                          in_size=args.embedding_dim, out_size=args.hidden_size_HANLayer,
                                          layer_num_heads=args.num_heads_HANLayer, dropout=args.dropout_HANLayer)
        elif args.dataset == 'yelp':
            self.HANLayer_user = HANLayer([['votes', 're-votes']],
                                          in_size=args.embedding_dim, out_size=args.hidden_size_HANLayer,
                                          layer_num_heads=args.num_heads_HANLayer, dropout=args.dropout_HANLayer)
        else:
            print('ERROR:class GraphEncodeModel_HAN2(nn.Module):')
            quit()

        self.HANLayer_item = HANLayer([['re-votes', 'votes']],
                                      in_size=args.embedding_dim, out_size=args.hidden_size_HANLayer,
                                      layer_num_heads=args.num_heads_HANLayer, dropout=args.dropout_HANLayer)
        self.snapAttention = SemanticAttention(in_size=args.hidden_size_HANLayer * args.num_heads_HANLayer)

    def forward(self, outputs, type_edges, combined_graph_data, hop_embedding):
        edges_embeddings = []

        all_embeddings = outputs.view(-1, self.args.embedding_dim)
        all_type_nodes = type_edges.view(-1, )  #
        user_embeddings = all_embeddings[all_type_nodes == 0]
        item_embeddings = all_embeddings[all_type_nodes == 1]
        condition = hop_embedding.view(-1, self.args.neighbour_num + 2) != 99
        ones = torch.ones(condition.shape, device=self.args.device)
        zeros = torch.zeros(condition.shape, device=self.args.device)
        gl_embedding = torch.where(condition, ones, zeros)

        hsg_combined = dgl.heterograph(combined_graph_data)
        hsg_combined.nodes['user'].data['feat'] = user_embeddings
        hsg_combined.nodes['item'].data['feat'] = item_embeddings

        time2 = time.time()
        hsg_combined_s = process_combined_hsg(hsg_combined).to(self.args.device)
        time3 = time.time()
        if item_embeddings.size(0) > 0:
            item_embeddings = self.HANLayer_item(hsg_combined, item_embeddings)
        if user_embeddings.size(0) > 0:
            user_embeddings = self.HANLayer_user(hsg_combined, user_embeddings)#这一行显存7146-》8478，应该是用户越多，显存越大，一下子显存用了好多
        time4 = time.time()

        # Restore embeddings to original positions
        combined_embeddings = torch.cat((user_embeddings, item_embeddings), dim=0)
        restored_embeddings = torch.zeros_like(combined_embeddings)
        restored_embeddings[all_type_nodes == 0] = user_embeddings
        restored_embeddings[all_type_nodes == 1] = item_embeddings

        time6 = time.time()

        node_count = self.args.neighbour_num + 2

        num_snapshots = outputs.size(0) * outputs.size(1)
        snapshot_indices = torch.arange(num_snapshots) * node_count
        end_indices = snapshot_indices + node_count

        # Create a tensor of start and end indices for all snapshots
        indices = torch.stack([snapshot_indices, end_indices], dim=1)

        snaps_embeddings1 = torch.stack([
            torch.mean(restored_embeddings[indices[i, 0]:indices[i, 1]], dim=0)#【】
            for i in range(num_snapshots)
        ]).to(self.args.device)

        snaps_tensor1 = snaps_embeddings1.view(outputs.size(0), self.args.window_size, -1)  # Add batch dimension
        edge_embedding1 = self.snapAttention(snaps_tensor1).squeeze()

        time7 = time.time()
        return edge_embedding1 if edge_embedding1.numel != 0 else torch.tensor([]), hsg_combined_s


class RelationalSubgraphEncoder2(nn.Module):
    def __init__(self, args):
        super(RelationalSubgraphEncoder2, self).__init__()
        self.args = args
        if args.dataset == 'digg_all':
            self.graph = dgl.heterograph({
                ('user', 'votes', 'item'): ([], []),
                ('item', 're-votes', 'user'): ([], []),
                ('user', 'trusts', 'user'): ([], []),
                ('user', 're-trusts', 'user'): ([], [])
            })

        elif args.dataset == 'amazon':
            self.graph = dgl.heterograph({
                ('user', 'votes', 'item'): ([], []),
                ('item', 're-votes', 'user'): ([], [])
            })
        elif args.dataset == 'yelp':
            self.graph = dgl.heterograph({
                ('user', 'votes', 'item'): ([], []),
                ('item', 're-votes', 'user'): ([], [])
            })
        else:
            print('ERROR:class RelationalSubgraphEncoder2(nn.Module)')
            quit()

        self.relational_encoding = RelationalNodeEncoding2(args=args, graph=self.graph)
        self.lstm = nn.ModuleDict({
            str(et): nn.GRU(args.hidden_units_rh * args.num_heads_rh, args.hidden_dim_GRU, batch_first=True).to(self.args.device)
            for et in self.graph.canonical_etypes
        })

        self.sample_nodes_num = [{etype: -1 for etype in self.graph.canonical_etypes} for _ in range(args.n_layers_rh)]
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler(self.sample_nodes_num)

    def forward(self, combined_hsg, mask_hsg_rs, hsg_edges_size0):
        device = self.args.device
        hsg_edges_size1 = self.args.window_size
        canonical_etypes = self.graph.canonical_etypes

        user_mask = mask_hsg_rs['user_mask']
        item_mask = mask_hsg_rs['item_mask']
        mask = mask_hsg_rs['mask']
        offsets = mask_hsg_rs['offsets']
        idx_num_all = mask_hsg_rs['idx_num_all']

        node_mask = {'user': user_mask, 'item': item_mask }
        time0 = time.time()
        node_embeddings_all = self.relational_encoding(hsg=combined_hsg, node_masks=node_mask)
        time1 = time.time()

        R_graph_embeddings_edges = []
        for i in range(hsg_edges_size0):
            R_grap_embeddings_edge = {}
            for etype in canonical_etypes:
                key = etype
                if mask[key][i][0] == 1:
                    snap_reps = []
                    for ii in range(hsg_edges_size1):
                        if mask[key][i][ii] == 1:
                            start = int(offsets[key][i][ii])
                            end = start + int(idx_num_all[key][i][ii])

                            snap_rep = torch.mean(node_embeddings_all[key][start:end], dim=0).squeeze()
                        else:
                            snap_rep = torch.zeros((self.args.hidden_units_rh * self.args.num_heads_rh),
                                                   device=device)

                        snap_reps.append(snap_rep)

                    Rsubgraph_change = torch.stack(snap_reps[::-1]).to(self.args.device)
                    R_grap_embeddings_edge[key] = Rsubgraph_change

            R_graph_embeddings_edges.append(R_grap_embeddings_edge)
        for etype in canonical_etypes:
            key = etype
            Rsubgraph_changes = [R_grap_embeddings_edge[key] for R_grap_embeddings_edge in R_graph_embeddings_edges if
                                 key in R_grap_embeddings_edge]

            if Rsubgraph_changes:
                Rsubgraph_changes = torch.stack(Rsubgraph_changes)
                _, lstm_output = self.lstm[str(key)](Rsubgraph_changes)
                lstm_output = lstm_output.permute(1, 0, 2)
                lstm_output_idx = 0
                for R_grap_embeddings_edge in R_graph_embeddings_edges:
                    if key in R_grap_embeddings_edge:
                        R_grap_embeddings_edge[key] = lstm_output[lstm_output_idx]
                        lstm_output_idx += 1
        return R_graph_embeddings_edges

    def _block_sampler(self, hsg, node_idx, ntype, device):
        loader = dgl.dataloading.DataLoader(hsg, {ntype: node_idx}, self.sampler, batch_size=node_idx.size(0),
                                            drop_last=False, device=device)

        for i, (input_nodes, output_nodes, blocks) in enumerate(loader):
            blocks = [block.to(device) for block in blocks]

        return blocks

class RelationalNodeEncoding2(nn.Module):
    def __init__(self, args, graph):
        super(RelationalNodeEncoding2, self).__init__()

        self.args = args
        self.graph = graph
        self.r_hgnn = RGNN(graph,
                             input_dim_dict={ntype: args.embedding_dim for ntype in graph.ntypes},
                             hidden_dim=args.hidden_units_rh, relation_input_dim=args.relation_hidden_units_rh,
                             relation_hidden_dim=args.relation_hidden_units_rh,
                             num_layers=args.n_layers_rh, n_heads=args.num_heads_rh, dropout=args.dropout_rh,
                             residual=args.residual_rh
                             )
        self.sample_nodes_num = [{etype: -1 for etype in graph.canonical_etypes} for _ in range(args.n_layers_rh)]
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler(self.sample_nodes_num)

    def forward(self, hsg: dgl.DGLHeteroGraph, node_masks=None):
        device = self.args.device
        user_idx = hsg.nodes('user').to(device)
        item_idx = hsg.nodes('item').to(device)
        time1 = time.time()
        if node_masks is not None:
            nodes_representation_user = self._encode_nodes(hsg, user_idx, 'user', device, masks=node_masks['user'])
            nodes_representation_item = self._encode_nodes(hsg, item_idx, 'item', device, masks=node_masks['item'])
        else:
            nodes_representation_user = self._encode_nodes(hsg, user_idx, 'user', device)
            nodes_representation_item = self._encode_nodes(hsg, item_idx, 'item', device)
        nodes_representation = nodes_representation_user
        nodes_representation.update(nodes_representation_item)
        time4 = time.time()
        return nodes_representation

    def _encode_nodes(self, hsg, node_idx, ntype, device, masks=None):
        if node_idx.size(0) == 0:
            return {}

        loader = dgl.dataloading.DataLoader(hsg, {ntype: node_idx}, self.sampler, batch_size=node_idx.size(0),
                                            drop_last=False, device=device)
        nodes_representation = {}
        if masks is not None:
            for mask in masks:
                for k,v in mask.items():
                    mask[k] = torch.tensor(v).to(self.args.device)

        for i, (input_nodes, output_nodes, blocks) in enumerate(loader):
            blocks = [block.to(device) for block in blocks]
            input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in
                              blocks[0].canonical_etypes}
            nodes_representation = self.r_hgnn(blocks, copy.deepcopy(input_features), mask=masks)#deepcopy使得显存从8487-》8780

        return nodes_representation

class RGNN(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim_dict: dict, hidden_dim: int, relation_input_dim: int,
                 relation_hidden_dim: int, num_layers: int, n_heads: int = 4,
                 dropout: float = 0.2, negative_slope: float = 0.2, residual: bool = True, norm: bool = False):

        super(RGNN, self).__init__()

        self.input_dim_dict = input_dim_dict
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.relation_input_dim = relation_input_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.residual = residual
        self.norm = norm

        # relation embedding dictionary
        self.relation_embedding = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(relation_input_dim, 1)) for etype in graph.etypes
        })

        # align the dimension of different types of nodes
        self.projection_layer = nn.ModuleDict({
            ntype: nn.Linear(input_dim_dict[ntype], hidden_dim * n_heads) for ntype in input_dim_dict
        })

        # each layer takes in the heterogeneous graph as input
        self.layers = nn.ModuleList()

        # for each relation_layer
        self.layers.append(
            RGNN_Layer(graph, hidden_dim * n_heads, hidden_dim, relation_input_dim, relation_hidden_dim, n_heads,
                         dropout, negative_slope, residual, norm))
        for _ in range(1, self.num_layers):
            self.layers.append(RGNN_Layer(graph, hidden_dim * n_heads, hidden_dim, relation_hidden_dim * n_heads,
                                            relation_hidden_dim, n_heads, dropout, negative_slope, residual, norm))

        # transformation matrix for target node representation under each relation
        self.node_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, hidden_dim, hidden_dim)) for etype in graph.etypes
        })

        # transformation matrix for relation representation
        self.relation_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, relation_hidden_dim, hidden_dim)) for etype in graph.etypes
        })

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')

        for etype in self.relation_embedding:
            nn.init.xavier_normal_(self.relation_embedding[etype], gain=gain)
        for ntype in self.projection_layer:
            nn.init.xavier_normal_(self.projection_layer[ntype].weight, gain=gain)
        for etype in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[etype], gain=gain)
        for etype in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[etype], gain=gain)

    def forward(self, blocks: list, relation_target_node_features: dict, relation_embedding: dict = None, mask = None):

        # target relation feature projection
        for stype, reltype, dtype in relation_target_node_features:
            relation_target_node_features[(stype, reltype, dtype)] = self.projection_layer[dtype](
                relation_target_node_features[(stype, reltype, dtype)])

        if relation_embedding is None:
            relation_embedding = {}
            for etype in self.relation_embedding:
                relation_embedding[etype] = self.relation_embedding[etype].flatten()

        # graph convolution
        i = 0
        for block, layer in zip(blocks, self.layers):
            if mask is not None:
                relation_target_node_features, relation_embedding = layer(block, relation_target_node_features,
                                                                          relation_embedding, mask[i])
            else:
                relation_target_node_features, relation_embedding = layer(block, relation_target_node_features,
                                                                          relation_embedding)

            i = i + 1

        return  relation_target_node_features


class RGNN_Layer(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim: int, hidden_dim: int, relation_input_dim: int,
                 relation_hidden_dim: int, n_heads: int = 8, dropout: float = 0.2, negative_slope: float = 0.2,
                 residual: bool = True, norm: bool = False):

        super(RGNN_Layer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relation_input_dim = relation_input_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.residual = residual
        self.norm = norm

        # node transformation parameters of each type
        self.node_transformation_weight = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(input_dim, n_heads * hidden_dim))
            for ntype in graph.ntypes
        })

        # relation transformation parameters of each type, used as attention queries
        self.relation_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(relation_input_dim, n_heads * 2 * hidden_dim))
            for etype in graph.etypes
        })

        # relation propagation layer of each relation
        self.relation_propagation_layer = nn.ModuleDict({
            etype: nn.Linear(relation_input_dim, n_heads * relation_hidden_dim)
            for etype in graph.etypes
        })

        # hetero conv modules, each RelationGraphConv deals with a single type of relation
        self.hetero_conv = HeteroGraphConv({
            etype: RelationGraphConv(in_feats=(input_dim, input_dim), out_feats=hidden_dim,
                                     num_heads=n_heads, dropout=dropout, negative_slope=negative_slope)
            for etype in graph.etypes
        })

        if self.residual:
            # residual connection
            self.res_fc = nn.ModuleDict()
            self.residual_weight = nn.ParameterDict()
            for ntype in graph.ntypes:
                self.res_fc[ntype] = nn.Linear(input_dim, n_heads * hidden_dim)
                self.residual_weight[ntype] = nn.Parameter(torch.randn(1))

        if self.norm:
            self.layer_norm = nn.ModuleDict({ntype: nn.LayerNorm(n_heads * hidden_dim) for ntype in graph.ntypes})

        # relation type crossing attention trainable parameters
        self.relations_crossing_attention_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, hidden_dim))
            for etype in graph.etypes
        })
        # different relations crossing layer
        self.relations_crossing_layer = RelationCrossing(in_feats=n_heads * hidden_dim,
                                                         out_feats=hidden_dim,
                                                         num_heads=n_heads,
                                                         dropout=dropout,
                                                         negative_slope=negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[weight], gain=gain)
        for weight in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[weight], gain=gain)
        for etype in self.relation_propagation_layer:
            nn.init.xavier_normal_(self.relation_propagation_layer[etype].weight, gain=gain)
        if self.residual:
            for ntype in self.res_fc:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)
        for weight in self.relations_crossing_attention_weight:
            nn.init.xavier_normal_(self.relations_crossing_attention_weight[weight], gain=gain)

    def forward(self, graph: dgl.DGLHeteroGraph, relation_target_node_features: dict, relation_embedding: dict, mask = None):

        input_src = relation_target_node_features

        if graph.is_block:
            input_dst = {}
            for srctype, etypye, dsttype in relation_target_node_features:
                input_dst[(srctype, etypye, dsttype)] = relation_target_node_features[(srctype, etypye, dsttype)][
                                                        :graph.number_of_dst_nodes(dsttype)]
        else:
            input_dst = relation_target_node_features

        output_features = self.hetero_conv(graph, input_src, input_dst, relation_embedding,
                                           self.node_transformation_weight, self.relation_transformation_weight)

        # residual connection for the target node
        if self.residual:
            for srctype, etype, dsttype in output_features:
                alpha = F.sigmoid(self.residual_weight[dsttype])
                output_features[(srctype, etype, dsttype)] = output_features[(srctype, etype, dsttype)] * alpha + \
                                                             self.res_fc[dsttype](
                                                                 input_dst[(srctype, etype, dsttype)]) * (1 - alpha)
        output_features_dict = {}

        # different relations crossing layer
        time0 = time.time()
        for srctype, etype, dsttype in output_features:
            if mask is not None:
                dst_node_relations_features = torch.stack([output_features[(stype, reltype, dtype)]
                                                   for stype, reltype, dtype in output_features if dtype == dsttype], dim=0)

                mask_t1 = torch.stack([mask[(stype, reltype, dtype)]
                                                   for stype, reltype, dtype in output_features.keys() if dtype == dsttype], dim=0)
                time2 = time.time()
                mask_t = fill_mask(mask_t1, self.n_heads).to(dst_node_relations_features.device)
                time3 = time.time()

                output_features_dict[(srctype, etype, dsttype)] = self.relations_crossing_layer(
                    dst_node_relations_features,
                    self.relations_crossing_attention_weight[etype],
                    mask=mask_t)
                time4 = time.time()
                output_features_dict[(srctype, etype, dsttype)] = output_features_dict[(srctype, etype, dsttype)] * \
                                                                  fill_mask1(mask[(srctype, etype, dsttype)], self.hidden_dim * self.n_heads).to(
                                                                      dst_node_relations_features.device)
                time5 = time.time()
            else:
                dst_node_relations_features = torch.stack([output_features[(stype, reltype, dtype)]
                     for stype, reltype, dtype in output_features if dtype == dsttype], dim=0)

                output_features_dict[(srctype, etype, dsttype)] = self.relations_crossing_layer(
                    dst_node_relations_features,
                    self.relations_crossing_attention_weight[etype],
                    mask=None)

        # layer norm for the output
        if self.norm:
            for srctype, etype, dsttype in output_features_dict:
                output_features_dict[(srctype, etype, dsttype)] = self.layer_norm[dsttype](output_features_dict[(srctype, etype, dsttype)])

        relation_embedding_dict = {}
        for etype in relation_embedding:
            relation_embedding_dict[etype] = self.relation_propagation_layer[etype](relation_embedding[etype])

        return output_features_dict, relation_embedding_dict

class HeteroGraphConv(nn.Module):


    def __init__(self, mods: dict):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)

    def forward(self, graph: dgl.DGLHeteroGraph, input_src: dict, input_dst: dict, relation_embedding: dict,
                node_transformation_weight: nn.ParameterDict, relation_transformation_weight: nn.ParameterDict):

        # find reverse relation dict
        reverse_relation_dict = {}
        for srctype, reltype, dsttype in list(input_src.keys()):
            for stype, etype, dtype in input_src:
                if stype == dsttype and dtype == srctype and etype != reltype:
                    reverse_relation_dict[reltype] = etype
                    break

        # dictionary, {(srctype, etype, dsttype): representations}
        outputs = dict()

        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                continue
            # for example, (author, writes, paper) relation, take author as src_nodes, take paper as dst_nodes
            dst_representation = self.mods[etype](rel_graph,
                                                  (input_src[(dtype, reverse_relation_dict[etype], stype)],
                                                   input_dst[(stype, etype, dtype)]),
                                                  node_transformation_weight[dtype],
                                                  node_transformation_weight[stype],
                                                  relation_embedding[etype],
                                                  relation_transformation_weight[etype])

            outputs[(stype, etype, dtype)] = dst_representation
        return outputs

class RelationGraphConv(nn.Module):

    def __init__(self, in_feats: tuple, out_feats: int, num_heads: int, dropout: float = 0.0, negative_slope: float = 0.2):
        super(RelationGraphConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats[0], in_feats[1]
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.relu = nn.ReLU()

    def forward(self, graph: dgl.DGLHeteroGraph, feat: tuple, dst_node_transformation_weight: nn.Parameter,
                src_node_transformation_weight: nn.Parameter, relation_embedding: torch.Tensor,
                relation_transformation_weight: nn.Parameter):
        graph = graph.local_var()
        # Tensor, (N_src, input_src_dim)
        feat_src = self.dropout(feat[0])
        # Tensor, (N_dst, input_dst_dim)
        feat_dst = self.dropout(feat[1])
        # Tensor, (N_src, n_heads, hidden_dim) -> (N_src, input_src_dim) * (input_src_dim, n_heads * hidden_dim)
        feat_src = torch.matmul(feat_src, src_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
        # Tensor, (N_dst, n_heads, hidden_dim) -> (N_dst, input_dst_dim) * (input_dst_dim, n_heads * hidden_dim)
        feat_dst = torch.matmul(feat_dst, dst_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
        # Tensor, (n_heads, 2 * hidden_dim) -> (1, input_dst_dim) * (input_dst_dim, n_heads * hidden_dim)
        relation_attention_weight = torch.matmul(relation_embedding.unsqueeze(dim=0), relation_transformation_weight).view(self._num_heads, 2 * self._out_feats)

        # first decompose the weight vector into [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j, This implementation is much efficient
        # Tensor, (N_dst, n_heads, 1),   (N_dst, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_dst = (feat_dst * relation_attention_weight[:, :self._out_feats]).sum(dim=-1, keepdim=True)
        # Tensor, (N_src, n_heads, 1),   (N_src, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_src = (feat_src * relation_attention_weight[:, self._out_feats:]).sum(dim=-1, keepdim=True)
        # (N_src, n_heads, hidden_dim), (N_src, n_heads, 1)
        graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
        # (N_dst, n_heads, 1)
        graph.dstdata.update({'e_dst': e_dst})
        # compute edge attention, e_src and e_dst are a_src * Wh_src and a_dst * Wh_dst respectively.
        graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))
        # shape (edges_num, heads, 1)
        e = self.leaky_relu(graph.edata.pop('e'))

        # compute softmax
        graph.edata['a'] = edge_softmax(graph, e)

        graph.update_all(fn.u_mul_e('ft', 'a', 'msg'), fn.sum('msg', 'feat'))
        # (N_dst, n_heads * hidden_dim), reshape (N_dst, n_heads, hidden_dim)
        dst_features = graph.dstdata.pop('feat').reshape(-1, self._num_heads * self._out_feats)

        dst_features = self.relu(dst_features)

        return dst_features

class RelationCrossing(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, num_heads: int, dropout: float = 0.0, negative_slope: float = 0.2):
        super(RelationCrossing, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dsttype_node_features: torch.Tensor, relations_crossing_attention_weight: nn.Parameter, mask=None):
        if len(dsttype_node_features) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            dsttype_node_features = dsttype_node_features.squeeze(dim=0)
        else:
            # (dsttype_node_relations_num, num_dst_nodes, n_heads, hidden_dim)
            dsttype_node_features = dsttype_node_features.reshape(dsttype_node_features.shape[0], -1, self._num_heads, self._out_feats)
            # shape -> (dsttype_node_relations_num, dst_nodes_num, n_heads, 1),  (dsttype_node_relations_num, dst_nodes_num, n_heads, hidden_dim) * (n_heads, hidden_dim)
            dsttype_node_relation_attention = (dsttype_node_features * relations_crossing_attention_weight).sum(dim=-1, keepdim=True)
            if mask is not None:
                dsttype_node_relation_attention = self.leaky_relu(dsttype_node_relation_attention) * (mask.unsqueeze(-1))
            else:
                dsttype_node_relation_attention = self.leaky_relu(dsttype_node_relation_attention)
            dsttype_node_relation_attention = F.softmax(dsttype_node_relation_attention, dim=0)
            # shape -> (dst_nodes_num, n_heads, hidden_dim),  (dsttype_node_relations_num, dst_nodes_num, n_heads, hidden_dim) * (dsttype_node_relations_num, dst_nodes_num, n_heads, 1)
            dsttype_node_features = (dsttype_node_features * dsttype_node_relation_attention).sum(dim=0)
            dsttype_node_features = self.dropout(dsttype_node_features)
            # shape -> (dst_nodes_num, n_heads * hidden_dim)
            dsttype_node_features = dsttype_node_features.reshape(-1, self._num_heads * self._out_feats)

        return dsttype_node_features

class RelationFusing(nn.Module):

    def __init__(self, node_hidden_dim: int, relation_hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        super(RelationFusing, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dst_node_features: list, dst_relation_embeddings: list,
                dst_node_feature_transformation_weight: list,
                dst_relation_embedding_transformation_weight: list):
        if len(dst_node_features) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            dst_node_relation_fusion_feature = dst_node_features[0]
        else:
            # (num_dst_relations, nodes, n_heads, node_hidden_dim)
            dst_node_features = torch.stack(dst_node_features, dim=0).reshape(len(dst_node_features), -1,
                                                                              self.num_heads, self.node_hidden_dim)
            # (num_dst_relations, n_heads, relation_hidden_dim)
            dst_relation_embeddings = torch.stack(dst_relation_embeddings, dim=0).reshape(len(dst_node_features),
                                                                                          self.num_heads,
                                                                                          self.relation_hidden_dim)
            # (num_dst_relations, n_heads, node_hidden_dim, node_hidden_dim)
            dst_node_feature_transformation_weight = torch.stack(dst_node_feature_transformation_weight, dim=0).reshape(
                len(dst_node_features), self.num_heads,
                self.node_hidden_dim, self.node_hidden_dim)
            # (num_dst_relations, n_heads, relation_hidden_dim, relation_hidden_dim)
            dst_relation_embedding_transformation_weight = torch.stack(dst_relation_embedding_transformation_weight,
                                                                       dim=0).reshape(len(dst_node_features),
                                                                                      self.num_heads,
                                                                                      self.relation_hidden_dim,
                                                                                      self.node_hidden_dim)
            # shape (num_dst_relations, nodes, n_heads, hidden_dim)
            dst_node_features = torch.einsum('abcd,acde->abce', dst_node_features,
                                             dst_node_feature_transformation_weight)

            # shape (num_dst_relations, n_heads, hidden_dim)
            dst_relation_embeddings = torch.einsum('abc,abcd->abd', dst_relation_embeddings,
                                                   dst_relation_embedding_transformation_weight)

            # shape (num_dst_relations, nodes, n_heads, 1)
            attention_scores = (dst_node_features * dst_relation_embeddings.unsqueeze(dim=1)).sum(dim=-1, keepdim=True)
            attention_scores = F.softmax(self.leaky_relu(attention_scores), dim=0)
            # (nodes, n_heads, hidden_dim)
            dst_node_relation_fusion_feature = (dst_node_features * attention_scores).sum(dim=0)
            dst_node_relation_fusion_feature = self.dropout(dst_node_relation_fusion_feature)
            # (nodes, n_heads * hidden_dim)
            dst_node_relation_fusion_feature = dst_node_relation_fusion_feature.reshape(-1,
                                                                                        self.num_heads * self.node_hidden_dim)

        return dst_node_relation_fusion_feature

class LandmarkMarkMatchs(nn.Module):
    def __init__(self, args):
        super(LandmarkMarkMatchs, self).__init__()
        self.args = args
        if args.dataset == 'digg_all':
            self.graph = dgl.heterograph({
                ('user', 'votes', 'item'): ([], []),
                ('item', 're-votes', 'user'): ([], []),
                ('user', 'trusts', 'user'): ([], []),
                ('user', 're-trusts', 'user'): ([], [])
            })

        elif args.dataset == 'amazon':
            self.graph = dgl.heterograph({
                ('user', 'votes', 'item'): ([], []),
                ('item', 're-votes', 'user'): ([], [])
            })
        elif args.dataset == 'yelp':
            self.graph = dgl.heterograph({
                ('user', 'votes', 'item'): ([], []),
                ('item', 're-votes', 'user'): ([], [])
            })
        else:
            print('ERROR:class LandmarkMarkMatchs(nn.Module):def init')
            quit()

        self.matchings = nn.ModuleDict({
            str(et): LandmarkMarkMatch(args=args).to(self.args.device)
            for et in self.graph.canonical_etypes
        })
        # self.matching = LandmarkMarkMatch(args=args).to(self.args.device)
        self.weighted_sum = nn.Linear(4, 1, bias=False)
        self.sematic_predict = SemanticAttention(args.hidden_dim_GRU)
        self.predict2 = nn.Linear(args.hidden_dim_GRU, 1)
        self.cal_sim_score = cal_sim_temp(args=args)

    def forward(self, R_graph_embeddings_edges, edges_embeddings, istest):
        if self.args.dataset == 'digg_all':
            change_embeddings = {
                ('user', 'votes', 'item'): [],
                ('item', 're-votes', 'user'): [],
                ('user', 'trusts', 'user'): [],
                ('user', 're-trusts', 'user'): []
            }
            mask = {
                ('user', 'votes', 'item'): [],
                ('item', 're-votes', 'user'): [],
                ('user', 'trusts', 'user'): [],
                ('user', 're-trusts', 'user'): []
            }

        elif self.args.dataset == 'amazon':
            change_embeddings = {
                ('user', 'votes', 'item'): [],
                ('item', 're-votes', 'user'): []
            }
            mask = {
                ('user', 'votes', 'item'): [],
                ('item', 're-votes', 'user'): []
            }
        elif self.args.dataset == 'yelp':
            change_embeddings = {
                ('user', 'votes', 'item'): [],
                ('item', 're-votes', 'user'): []
            }
            mask = {
                ('user', 'votes', 'item'): [],
                ('item', 're-votes', 'user'): []
            }
        else:
            print('ERROR:class LandmarkMarkMatchs(nn.Module):def forward')
            quit()

        for key in change_embeddings.keys():
            for R_graph_embeddings_edge in R_graph_embeddings_edges:
                if key in R_graph_embeddings_edge:
                    change_embeddings[key].append(R_graph_embeddings_edge[key])
                    mask[key].append(True)
                else:
                    change_embeddings[key].append(torch.zeros((1, self.args.hidden_dim_GRU), device=self.args.device))
                    mask[key].append(False)
        change_embeddings_t = torch.stack([torch.stack(change_embeddings[key]) for key in change_embeddings.keys()]).contiguous().squeeze()
        masked = torch.tensor([mask[key] for key in mask.keys()]).to(self.args.device)
        mask_sim = self.cal_sim_score(change_embeddings=change_embeddings_t, edges_embeddings= edges_embeddings, mask=masked)


        change_socres, change_socres_eb, sim_cs, zloss = {}, {}, {}, {}
        for key in change_embeddings.keys():
            change_socres[key], change_socres_eb[key], sim_cs[key], zloss[key] = self.matchings[str(key)](torch.stack(change_embeddings[key]).squeeze(),istest)

        temp1 = torch.stack([torch.stack([change_socres_eb[key] for key in change_socres_eb.keys()])]).contiguous().squeeze().transpose(0, 1)
        mask_eb = masked.int().unsqueeze(-1).expand(-1, -1, self.args.hidden_dim_GRU).transpose(0, 1)
        predict = self.sematic_predict(mask_eb * temp1)
        predict = self.predict2(predict)
        zloss_return = torch.stack(
            [zloss[key] for key in zloss.keys()]).contiguous().squeeze().mean() / edges_embeddings.size(0)

        return predict, sim_cs, zloss_return


class cal_sim_temp(nn.Module):
    def __init__(self, args):
        super(cal_sim_temp, self).__init__()
        self.args = args
        heads = 8
        dim_heads = 125
        self.heads = heads
        self.dim_heads = dim_heads
        self.inner_dim = 128
        self.to_k = nn.Linear(args.hidden_size_HANLayer * args.num_heads_HANLayer, self.inner_dim, bias=False)
        self.to_q = nn.Linear(args.hidden_dim_GRU, self.inner_dim, bias=False)

    def forward(self, change_embeddings, edges_embeddings, mask):
        k = self.to_k(edges_embeddings)
        sim_all = []
        for i in range(change_embeddings.size(0)):
            q = self.to_q(change_embeddings[i])


            sim = einsum('i d,j d -> i j', q, k) * (self.inner_dim ** -0.5)
            sim = torch.diag(sim)

            sim_all.append(sim)
        sim_all = torch.stack(sim_all)
        mask_sim = sim_all.masked_fill(mask == False,-1e9)
        mask_sim = F.softmax(mask_sim, dim=0)
        return mask_sim.transpose(0, 1)



class LandmarkMarkMatch(nn.Module):
    def __init__(self, args):
        super(LandmarkMarkMatch, self).__init__()
        self.kv_dim = args.heads_M * args.inner_dim_M
        self.X_dim = args.hidden_dim_GRU
        self.M_dim = args.M_dim_M
        self.inner_dim = args.inner_dim_M
        self.percent_selected = args.percent_selected_M
        self.lay_number = args.layers_num_M
        self.cross_layers = M_cross_layers(num_layers=args.cross_MLP_layers_M, M_dim=args.M_dim_M,
                                           heads=args.heads_M, dim_heads=args.inner_dim_M,
                                           dropout=args.dropout_M)
        self.to_kv = nn.Linear(self.X_dim, self.kv_dim*2, bias=False)
        self.Memory = nn.Parameter(torch.randn(args.memory_num, self.M_dim))

    def forward(self, data: torch.tensor, istest):
        M = self.Memory.clone()  # 克隆Memory，避免直接修改
        sim_M = calucate_sim(self.X_dim, self.M_dim, self.inner_dim, data, M)
        percent_count = int(data.size(0) * self.percent_selected)
        _, top_indices = torch.topk(sim_M, percent_count)

        if not istest:
            X = torch.zeros_like(data).to(data.device)
            X[top_indices] = data[top_indices]


            for i in range(self.lay_number):
                M = self.cross_layers(M=M, X=X, to_kv=self.to_kv)
                sim_M = calucate_sim(self.X_dim, self.M_dim, self.inner_dim, data, M)
                _, top_indices = torch.topk(sim_M, percent_count)

                X = torch.zeros_like(data).to(data.device)
                X[top_indices] = data[top_indices]


        out_ebs, result_loss, zloss = calucate_emb(self.X_dim, self.M_dim, self.inner_dim, data, M)

        return sim_M, out_ebs, result_loss, zloss



class M_cross_layers(nn.Module):
    def __init__(self,num_layers,M_dim,heads=8, dim_heads=64,dropout=0.0):
        super(M_cross_layers, self).__init__()
        self.num_layers = num_layers
        self.M_dim = M_dim

        self.heads = heads
        self.dim_heads = dim_heads

        self.layer_stack = nn.ModuleList([
            cross_layer(M_dim,heads, dim_heads, dropout)
            for _ in range(num_layers)])

    def forward(self, M, X,to_kv):
        '''块内不参数共享'''

        for layer in self.layer_stack:
            output =layer(M,X,to_kv)
            M=output

        return M


class cross_layer(nn.Module):
    def __init__(self,M_dim,heads, dim_heads, dropout):
        super(cross_layer, self).__init__()
        self.self_attn = cross_att(M_dim,heads, dim_heads)
        self.feed_forward =FeedForward( M_dim, mult=4,dropout=0.1)
        self.sublayer = clones(SublayerConnection(dropout), 2)


    def forward(self,M, x,to_kv):
        # attention sub layer
        m = self.sublayer[0](M,x,to_kv,self.self_attn)

        return m

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class cross_att(nn.Module):
    def __init__(self, M_dim, heads, dim_heads):
        super(cross_att, self).__init__()
        self.M_dim = M_dim
        self.heads = heads
        self.dim_heads = dim_heads

        self.inner_dim = dim_heads * heads

        self.to_q =nn.Linear(M_dim, self.inner_dim, bias=False)

        self.mutl_heads =nn.Linear(self.inner_dim, M_dim)

    def forward(self, M, X,to_kv):


        M, X = map(lambda t: repeat(t, 'n d ->b n d', b=1), (M, X))  # 添加维度 1 x e x c / 1 x node_num x d

        q = self.to_q(M)  # 1 x e x inner
        k, v = to_kv(X).chunk(2, dim=-1)  # 1 x node_num x inner

        q, k, v = map(lambda t: rearrange(t, 'b n (h dim) -> (b h) n dim', h=self.heads),
                      (q, k, v))  # h x n x dim_heads

        sim = einsum('b i d,b j d -> b i j', q, k) * (self.dim_heads ** -0.5)

        mask_sim=F.softmax(sim, dim=-1)

        out_M = einsum('b i j, b j d -> b i d', mask_sim, v)
        out_M = rearrange(out_M, '(b h) n d -> b n (h d)', h=self.heads)
        out_M=out_M.squeeze(0)

        out=self.mutl_heads(out_M)

        return out

class SublayerConnection(nn.Module):

    def __init__(self, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, M,x,to_kv,layer):

        out_nodes = layer(M, x, to_kv)

        out_nodes = self.dropout(out_nodes)
        if out_nodes.shape[-1] == M.shape[-1]:
            out_nodes += M
        else:
            raise ValueError('args.dim_heads改成和M_dim一致')

        out_nodes = F.layer_norm(out_nodes, [out_nodes.size()[-1]])


        return out_nodes

class FeedForward(nn.Module):
    def __init__(self, M_dim, mult=4,dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(M_dim, M_dim * mult)
        self.w_2 = nn.Linear(M_dim * mult, M_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = BatchNorm1d(M_dim * mult)

    def forward(self, out,N1,N2):

        return self.w_2(self.dropout(F.relu(self.w_1(out))))
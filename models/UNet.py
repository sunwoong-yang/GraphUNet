import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GMMConv, pool, knn_interpolate, GraphNorm, LayerNorm, ChebConv, GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from lib import pooling
import time
import warnings
import gc
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')


"""
Pooling & unpooling is largely based on Graph-UNets
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/graph_unet.html
"""

class Encoder(torch.nn.Module):

    def __init__(self, HyperParams):
        super().__init__()
        self.HyperParams = HyperParams
        self.hidden_channels = HyperParams.Enc_HC
        self.depth = len(self.hidden_channels)
        self.act = HyperParams.act
        self.skip = HyperParams.skip
        self.input_size = HyperParams.num_nodes # 2719
        self.pooling_layer = HyperParams.pooling
        self.beta = HyperParams.beta

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for i in range(self.depth):

            if i != self.depth -1 :
                if HyperParams.gcn_type == 0:
                    self.down_convs.append(GMMConv(self.hidden_channels[i], self.hidden_channels[i+1], dim=1, kernel_size=HyperParams.gcn_k))
                elif HyperParams.gcn_type == 1:
                    self.down_convs.append(ChebConv(self.hidden_channels[i], self.hidden_channels[i+1], K=HyperParams.gcn_k))
                elif HyperParams.gcn_type in [2, 5]:
                    self.down_convs.append(GCNConv(self.hidden_channels[i], self.hidden_channels[i+1], improved=False))
                elif HyperParams.gcn_type == 3:
                    self.down_convs.append(GATConv(self.hidden_channels[i], self.hidden_channels[i+1], heads=HyperParams.gcn_k, concat=False))
                elif HyperParams.gcn_type in [4, 6]:
                    self.down_convs.append(GCNConv(self.hidden_channels[i], self.hidden_channels[i+1], improved=True))

                if self.HyperParams.Norm in ["G", "g"]:
                    self.norms.append(GraphNorm(self.hidden_channels[i+1]))
                elif self.HyperParams.Norm in ["L", "l"]:
                    self.norms.append(LayerNorm(self.hidden_channels[i+1]))

            else:
                if HyperParams.gcn_type == 0:
                    self.down_convs.append(GMMConv(self.hidden_channels[i], self.hidden_channels[i], dim=1, kernel_size=HyperParams.gcn_k))
                elif HyperParams.gcn_type == 1:
                    self.down_convs.append(ChebConv(self.hidden_channels[i], self.hidden_channels[i], K=HyperParams.gcn_k))
                elif HyperParams.gcn_type in [2, 5]:
                    self.down_convs.append(GCNConv(self.hidden_channels[i], self.hidden_channels[i], improved=False))
                elif HyperParams.gcn_type == 3:
                    self.down_convs.append(GATConv(self.hidden_channels[i], self.hidden_channels[i], heads=HyperParams.gcn_k, concat=False))
                elif HyperParams.gcn_type in [4, 6]:
                    self.down_convs.append(GCNConv(self.hidden_channels[i], self.hidden_channels[i], improved=True))

                if self.HyperParams.Norm in ["G", "g"]:
                    self.norms.append(GraphNorm(self.hidden_channels[i]))
                elif self.HyperParams.Norm in ["L", "l"]:
                    self.norms.append(LayerNorm(self.hidden_channels[i]))

        # if HyperParams.gcn_type == 0:
        #     self.down_convs.append(GMMConv(self.hidden_channels[self.depth - 1], self.hidden_channels[self.depth - 1], dim=1, kernel_size=5))
        # elif HyperParams.gcn_type == 1:
        #     self.down_convs.append(ChebConv(self.hidden_channels[self.depth - 1], self.hidden_channels[self.depth - 1], K=2))
        # elif HyperParams.gcn_type == 2:
        #     self.down_convs.append(GCNConv(self.hidden_channels[self.depth - 1], self.hidden_channels[self.depth - 1]))

        if self.pooling_layer is not None:
            for i in range(len(self.pooling_layer)):
                if self.HyperParams.my_pooling:
                    self.pools.append(pooling.TopKPooling(self.hidden_channels[i + 1], ratio=self.pooling_layer[i]))
                else:
                    self.pools.append(pool.TopKPooling(self.hidden_channels[i+1], ratio=self.pooling_layer[i]))
                 # self.pools.append(pooling.TopKPooling(self.hidden_channels[i + 1], ratio=self.pooling_layer[i]))




        self.reset_parameters()

    def encoder(self, data, if_plot_mesh=None):

        x = data.x
        edge_index = data.edge_index  # shape(2, 5235840)
        if if_plot_mesh is not None:
            self.plot_mesh(None, None, data, pool_step=0, repeat_idx=if_plot_mesh)
        if self.HyperParams.gcn_type in [5, 6]:
            edge_weight = x.new_ones(edge_index.size(1))
        else:
            edge_weight = data.edge_attr # shape(5235840=480*10908)



        # Pooling procedure
        if self.pooling_layer is not None:

            # xs = [x]
            # edge_indices = [edge_index]
            # edge_weights = [edge_weight]
            # perms = []
            # batch = data.batch

            for layer_idx, (conv_layer, pool_layer) in enumerate(zip(self.down_convs, self.pools)):

                conv_x = conv_layer(x, edge_index, edge_weight.unsqueeze(1))  # x: (0, 288000) torch.Size([92446, 1])

                # if layer_idx == 0:
                #     batch = data.batch

                if self.HyperParams.Norm in ["G","g","L","l"]:
                    if layer_idx == 0:
                        batch = data.batch
                    conv_x = self.norms[layer_idx](conv_x, batch)

                # Activation function is applied after GraphNorm
                # https://github.com/lsj2408/GraphNorm/blob/master/GraphNorm_ws/gnn_ws/gnn_example/model/GCN/gcn_all.py
                conv_x = self.act(conv_x)


                if layer_idx == 0:
                    xs = [conv_x]
                    edge_indices = [edge_index]
                    edge_weights = [edge_weight]
                    perms = []
                    batches = [data.batch]
                    batch = data.batch
                    batches4plot_mesh = []
                else :
                    xs.append(conv_x)
                    edge_indices.append(edge_index)
                    edge_weights.append(edge_weight)
                    batches.append(pooled_batch)

                # if self.HyperParams.augmentation_order:
                edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, conv_x.size(0), order = self.HyperParams.augmentation_order)

                pooled_x, pooled_edge_index, pooled_edge_attr, pooled_batch, pooled_perm, pooled_score = \
                    pool_layer(conv_x, edge_index=edge_index, edge_attr=edge_weight, batch=batch) #torch.Size([34000, 1])
                perms.append(pooled_perm)
                batches4plot_mesh.append(pooled_batch)
                # print('idxidx',pooled_perm, torch.max(pooled_edge_index), torch.max(pooled_perm), pooled_edge_index.shape, pooled_batch.shape, pooled_perm.shape, pooled_x.shape)

                if if_plot_mesh is not None:
                    self.plot_mesh(batches4plot_mesh, perms, data, pool_step=layer_idx + 1, repeat_idx=if_plot_mesh)
                # batches.append(pooled_batch
                x = pooled_x
                edge_index = pooled_edge_index
                edge_weight = pooled_edge_attr
                batch = pooled_batch

            # Last conv btw the encoder and decoder
            conv_x = self.down_convs[-1](x, edge_index, edge_weight.unsqueeze(1))

            if self.HyperParams.Norm in ["G", "g", "L", "l"]:
                conv_x = self.norms[-1](conv_x, batch)
            x = self.act(conv_x)

            pooling_info = (xs, edge_indices, edge_weights, perms, batches)

        else:
            for layer_idx, layer in enumerate(self.down_convs):
                x = layer(x, edge_index, edge_weight.unsqueeze(1)) #x: (0, 288000) torch.Size([92446, 1])
                if layer_idx == 0:
                    batch = data.batch
                if self.HyperParams.Norm in ["G","g","L","l"]:
                    # if layer_idx == 0:
                    #     batch = data.batch
                    x = self.norms[layer_idx](x, batch)
                x = self.act(x)

                if layer_idx == 0:
                    xs = [x]
                else :
                    xs.append(x)
            pooling_info = (xs, edge_index, edge_weight, batch)

        #
        # if self.pooling_layer is not None:
        #     pooling_info = (xs, edge_indices, edge_weights, perms, batches)
        #     # print('pppp', xs[0].shape, xs[1].shape)
        # else:
        #     pooling_info = None
        # x = x.reshape(data.num_graphs, self.input_size * self.hidden_channels[-1])
        # x = self.act(self.fc_in1(x))
        # x = self.fc_in2(x)
        return x, pooling_info

    def reset_parameters(self):
        for model in self.down_convs:
            model.reset_parameters()
        for model in self.pools:
            model.reset_parameters()
        if self.HyperParams.Norm in ["G","g","L","l"]:
            for model in self.norms:
                model.reset_parameters()

            # for name, param in conv.named_parameters():
            #     if 'bias' in name:
            #         nn.init.constant_(param, 0)
            #     else:
            #         nn.init.kaiming_uniform_(param)


    def forward(self, data, if_plot_mesh=None):
        x, pooling_info = self.encoder(data, if_plot_mesh=if_plot_mesh)
        return x, pooling_info

    def augment_adj(self, edge_index, edge_weight, num_nodes, order):
        if order != 1:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                     num_nodes=num_nodes)
            # GPT version
            adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))

            for _ in range(order - 1):
                # Sparse matrix multiplication
                adj = torch.sparse.mm(adj, adj)
            # adj = to_torch_csr_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))
            # for _ in range(order - 1):
            #     adj = adj @ adj
            # adj = adj.to_sparse_coo()
            # adj = (adj @ adj).to_sparse_coo()
            edge_index, edge_weight = adj.indices(), adj.values()
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    # def plot_mesh(self, pos, edge_index, batch):
    def plot_mesh(self, batch, perm, data, pool_step, repeat_idx):
        xy = data.pos.cpu() #data.pos
        if (batch is None) or (perm is None):
            node_x = xy[:, 0]
            node_y = xy[:, 1]
            # import numpy as np
            # sliced_parts = np.concatenate((np.array(range(0, 29)), np.array(range(57,500))))
            # node_x = xy[sliced_parts, 0]
            # node_y = xy[sliced_parts, 1]
        else:
            for batch_ele, perm_ele in zip(batch, perm):
                batch_temp = [i == 0 for i in batch_ele]
                xy = xy[perm_ele[batch_temp].cpu()]
            node_x = xy[:, 0]
            node_y = xy[:, 1]
            # node_x = xy[-50:, 0]
            # node_y = xy[-50:, 1]

        fig, ax = plt.subplots()
        ax.scatter(node_x, node_y, s=0.5, c='k')
        # for i in range(edge_index.shape[0]):
        #     ax.plot((xy[edge_index[0, i], 0], xy[edge_index[1, i], 0]), (xy[edge_index[0, i], 1], xy[edge_index[1, i], 1]),
        #             color='k', lw=0.5)
        plt.tight_layout()
        plt.xlim(0, 1.6)
        plt.ylim(0, 0.4)
        ax.set_aspect('equal', 'box')
        # ax.set_title('Error field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
        if not os.path.exists(self.HyperParams.net_dir + '/figures'):
            os.makedirs(self.HyperParams.net_dir + '/figures')
        plt.savefig(self.HyperParams.net_dir + '/figures/' + f'Pooled{repeat_idx}_poolstep{pool_step}' + '.png', bbox_inches='tight',
                    dpi=500)
        plt.close()

class Decoder(torch.nn.Module):

    def __init__(self, HyperParams):
        super().__init__()

        self.HyperParams = HyperParams
        # self.hidden_channels = HyperParams.hidden_channels[::-1]
        # self.hidden_channels = [1]*len(HyperParams.hidden_channels) # 모든 hc를 1로 통일
        # self.hidden_channels = [1,5,10,5,1]  # ori
        self.hidden_channels = HyperParams.Dec_HC
        self.depth = len(self.hidden_channels)
        self.act = HyperParams.act
        self.skip = HyperParams.skip
        self.input_size = HyperParams.num_nodes
        if HyperParams.pooling is not None:
            self.pooling_layer = HyperParams.pooling[::-1]
        else:
            self.pooling_layer = None

        self.up_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        # Initialize GMMConv and TopKPooling layers

        for i in range(len(self.hidden_channels) - 1):
            # self.up_convs.append(GMMConv(self.hidden_channels[i], self.hidden_channels[i + 1], dim=1, kernel_size=5))
            if HyperParams.gcn_type == 0:
                self.up_convs.append(GMMConv(self.hidden_channels[i], self.hidden_channels[i+1], dim=1, kernel_size=HyperParams.gcn_k))
            elif HyperParams.gcn_type == 1:
                self.up_convs.append(ChebConv(self.hidden_channels[i], self.hidden_channels[i+1], K=HyperParams.gcn_k))
            elif HyperParams.gcn_type in [2,5]:
                self.up_convs.append(GCNConv(self.hidden_channels[i], self.hidden_channels[i+1], improved=False))
            elif HyperParams.gcn_type == 3:
                self.up_convs.append(GATConv(self.hidden_channels[i], self.hidden_channels[i+1], heads=HyperParams.gcn_k, concat=False))
            elif HyperParams.gcn_type in [4,6]:
                self.up_convs.append(GCNConv(self.hidden_channels[i], self.hidden_channels[i+1], improved=True))

            if self.HyperParams.Norm in ["G","g"]:
                self.norms.append(GraphNorm(self.hidden_channels[i + 1]))
            elif self.HyperParams.Norm in ["L", "l"]:
                self.norms.append(LayerNorm(self.hidden_channels[i + 1]))


        #for additional GCN (N -> 1)
        # self.last_up_convs = GMMConv(self.hidden_channels[-1], 1, dim=1, kernel_size=5)
        # if self.pooling_layer is not None:
        #     for i in range(len(self.pooling_layer)):
        #         if self.HyperParams.my_pooling:
        #             print('myyyy')
        #             self.pools.append(pooling.TopKPooling(self.hidden_channels[::-1][i+1], ratio=self.pooling_layer[::-1][i]))
        #         else:
        #             print('nooooo')
        #             self.pools.append(pool.TopKPooling(self.hidden_channels[::-1][i+1], ratio=self.pooling_layer[::-1][i]))


        self.reset_parameters()


    def decoder(self, x, pooling_info):
        """
        Decodes the input data.

        Parameters:
        - x: The encoded feature representations.
        - data: The graph data.

        Returns:
        Decoded output data.
        """
        if self.pooling_layer is not None:
            xs, edge_indices, edge_weights, perms, batches = pooling_info


            for idx, (layer, res, edge_index, edge_weight, perm, batch) in enumerate(
                    zip(self.up_convs, xs[::-1], edge_indices[::-1], edge_weights[::-1], perms[::-1], batches[::-1])):

                # Unpooling
                up = torch.zeros_like(res[:,[-1]]) # Initialize 'up' to be zeros with the size of total nodes at this iteration
                up = up.repeat(1, self.hidden_channels[idx]) # Make 'up' to have same channel # with 'x'
                up[perm] = x # Fill 'up' by 'x', where the indices are noted by 'perm'
                if self.HyperParams.zero_unpooling is False:
                    x_mean = torch.mean(x, dim=0)
                    mask = torch.ones(up.size(0), dtype=torch.bool)
                    mask[perm] = False
                    up[mask] = x_mean
                    # for i in range(len(up)):
                    #     if i not in perm:
                    #         up[i] = torch.mean(x)
                # up[perm] = x  # Fill 'up' by 'x', where the indices are noted by 'perm'

                if self.HyperParams.UNet:
                    up += res[:, [-1]] # Skip-connection for UNet using the last channel's x values

                conv_x = layer(up, edge_index, edge_weight.unsqueeze(1))

                if self.HyperParams.Norm in ["G","g","L","l"]:
                    conv_x = self.norms[idx](conv_x, batch)

                # Activation is not applied on the last layer
                if idx != len(self.up_convs) - 1:
                    conv_x = self.act(conv_x)

                x = conv_x

        else:

            xs, edge_index, edge_weight, batch = pooling_info
            for idx, (layer, res) in enumerate(zip(self.up_convs, xs[::-1])):
                if self.HyperParams.UNet:
                    x += res[:, [-1]] # Skip-connection for UNet using the last channel's x values

                conv_x = layer(x, edge_index, edge_weight.unsqueeze(1))

                if self.HyperParams.Norm in ["G","g","L","l"]:
                    conv_x = self.norms[idx](conv_x, batch)

                # Activation is not applied on the last layer
                if idx != len(self.up_convs) - 1:
                    conv_x = self.act(conv_x)

                x = conv_x
        # for additional GCN (N -> 1)
        # x = self.last_up_convs(x, edge_index, edge_weight.unsqueeze(1))
        # x = x.reshape()
        # print('last',xs[0].shape)
        return x


    def reset_parameters(self):
        for model in self.up_convs:
            model.reset_parameters()
        # for model in self.pools:
        #     model.reset_parameters()
        if self.HyperParams.Norm in ["G","g","L","l"]:
            for model in self.norms:
                model.reset_parameters()
        # self.last_up_convs.reset_parameters()

    def forward(self, x, data):
        x = self.decoder(x, data)
        return x

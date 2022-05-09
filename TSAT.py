import copy
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import xavier_normal_small_init_, xavier_uniform_small_init_

DEFAULT_LAMBDAS_ATTENTION = 0.3
DEFAULT_LAMBDAS_IMF_1 = 0.0999
DEFAULT_LAMBDAS_IMF_2 = 0.0801
DEFAULT_LAMBDAS_IMF_3 = 0.06
DEFAULT_LAMBDAS_IMF_4 = 0.0399
DEFAULT_LAMBDAS_IMF_5 = 0.0201
DEFAULT_LAMBDAS_ADJACENCY = 1. - DEFAULT_LAMBDAS_ATTENTION - DEFAULT_LAMBDAS_IMF_1 - DEFAULT_LAMBDAS_IMF_2 - DEFAULT_LAMBDAS_IMF_3 - DEFAULT_LAMBDAS_IMF_4 - DEFAULT_LAMBDAS_IMF_5
DEFAULT_LAMBDAS = (
    DEFAULT_LAMBDAS_ATTENTION, 
    DEFAULT_LAMBDAS_IMF_1,
    DEFAULT_LAMBDAS_IMF_2,
    DEFAULT_LAMBDAS_IMF_3,
    DEFAULT_LAMBDAS_IMF_4,
    DEFAULT_LAMBDAS_IMF_5,
    DEFAULT_LAMBDAS_ADJACENCY
    )


# Create the TSAT Model
def make_TSAT_model(l_backcast, n_nodes, d_edge=5, N=2, d_model=128, h=8, dropout=0.1,
                    lambda_attention=DEFAULT_LAMBDAS_ATTENTION, 
                    lambda_imf_1=DEFAULT_LAMBDAS_IMF_1,
                    lambda_imf_2=DEFAULT_LAMBDAS_IMF_2, 
                    lambda_imf_3=DEFAULT_LAMBDAS_IMF_3, 
                    lambda_imf_4=DEFAULT_LAMBDAS_IMF_4,
                    lambda_imf_5=DEFAULT_LAMBDAS_IMF_5,
                    trainable_lambda=False,
                    N_dense=2, leaky_relu_slope=0.0, aggregation_type='mean', 
                    dense_output_nonlinearity='relu', imf_matrix_kernel='softmax',
                    n_output=1,
                    scale_norm=False, init_type='uniform', use_adapter=False, n_generator_layers=1,**kwargs):
    """
    Construct a TSAT model from input hypterparameters.

    Detail:
    - Time_Series_Multihead_Attention block: the block that implement the attention mechanism
    - Sequential_FFN

    Core Components:
    1. Time_Embeddings
    2. Time_Series_Self_Attention
    3. Graph_Representation_Readout

    :return: model
    """
    TSMHA = Time_Series_Multi_Head_Attention(h, d_model, dropout, lambda_attention, 
                                            lambda_imf_1, lambda_imf_2, lambda_imf_3, lambda_imf_4, lambda_imf_5,
                                            trainable_lambda, imf_matrix_kernel
                                            )
    SFFN = Sequential_FFN(d_model, N_dense, dropout, leaky_relu_slope, dense_output_nonlinearity)
    model = Time_Series_Attention_Transformer(
        Time_Embeddings(
            d_model, l_backcast, dropout
            ),
        Time_Series_Self_Attention(
            Self_Attention_Block(
                d_model, copy.deepcopy(TSMHA), copy.deepcopy(SFFN), dropout, scale_norm, use_adapter
                ),
            N,
            scale_norm
            ),
        Graph_Representation_Readout(
            n_nodes, d_model, aggregation_type, n_output, n_generator_layers, leaky_relu_slope, dropout, scale_norm
            )
    )

    # Initialize the model parameters and weights with Glorot Initialization / Fan_avg Initialization.
    if init_type == 'normal':
        initialization_fun = nn.init.xavier_normal_
    elif init_type == 'uniform':
        initialization_fun = nn.init.xavier_uniform_
    elif init_type == 'constant':
        initialization_fun = nn.init.constant_
    elif init_type == 'orthogonal':
        initialization_fun = nn.init.orthogonal_
    elif init_type == 'small_normal_init':
        initialization_fun = xavier_normal_small_init_
    elif init_type == 'small_uniform_init':
        initialization_fun = xavier_uniform_small_init_
    else:
        raise ValueError(f'Invalid initialization type: {init_type}')
    for para in model.parameters():
        if para.dim() > 1:
            initialization_fun(para)
    
    return model



# =======================================================================================
### Main algorithm of TSAT
class Time_Series_Attention_Transformer(nn.Module):
    """
    Main part of TSAT model, it consists of time_embeddings, time_series_self_attention, and graph_representation_readout.
    """
    def __init__(self, time_embeddings, time_series_self_attention, graph_representation_readout):
        super(Time_Series_Attention_Transformer, self).__init__()
        self.time_embeddings = time_embeddings
        self.time_series_self_attention = time_series_self_attention
        self.graph_representation_readout = graph_representation_readout
    
    def _time_embeddings(self, multi_time_series):
        return self.time_embeddings(multi_time_series)
    
    def _time_series_self_attention(self, multi_time_series, ts_mask, adj_matrix, imf_1_matrix, imf_2_matrix, imf_3_matrix, imf_4_matrix,
                imf_5_matrix):
        return self.time_series_self_attention(self._time_embeddings(multi_time_series), ts_mask, adj_matrix, imf_1_matrix, imf_2_matrix, 
                            imf_3_matrix, imf_4_matrix, imf_5_matrix)
    
    def _graph_representation_readout(self, out, out_mask):
        return self.graph_representation_readout(out, out_mask)
    
    def forward(self, multi_time_series, ts_mask, adj_matrix, imf_1_matrix, imf_2_matrix, imf_3_matrix, imf_4_matrix,
                imf_5_matrix):
        """Take in and process masked src and target sequences."""
        x = self._time_series_self_attention(multi_time_series, ts_mask, adj_matrix, imf_1_matrix, imf_2_matrix, imf_3_matrix, imf_4_matrix, imf_5_matrix)
        return self._graph_representation_readout(x, ts_mask)



# =======================================================================================
### 1. Time Embeddings layer
class Time_Embeddings(nn.Module):
    def __init__(self, d_model, l_backcast, dropout):
        super(Time_Embeddings, self).__init__()
        self.linear = nn.Linear(l_backcast, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear(x))


# class Time_Embeddings(nn.Module):
#     def __init__(self, d_model, l_backcast, n_nodes, dropout, n_RNN_layers=1):
#         super(Time_Embeddings, self).__init__()
#         self.l_backcast = l_backcast
#         self.hidden_dimensions = d_model
#         self.n_hidden_dimensions = n_RNN_layers
#         self.RNN = nn.RNN(
#             input_size=l_backcast, 
#             hidden_size=d_model, 
#             num_layers=n_RNN_layers
#             )
#         self.dropout = nn.Dropout(dropout)
#         self.h_in = nn.Parameter(torch.zeros(self.n_hidden_dimensions, 
#                                               n_nodes, 
#                                               d_model),
#                                   requires_grad=True
#                                   )
#
#     def forward(self, x):
#         out, h_out = self.RNN(x, self.h_in)
#         return self.dropout(out)



# =======================================================================================
### 2. Time Series Self Attention layer
class Time_Series_Self_Attention(nn.Module):
    """
    Core Time Series Self-Attention block is a stack of N self-attention block
    """
    def __init__(self, layer, N, scale_norm:bool):
        super(Time_Series_Self_Attention, self).__init__()
        assert isinstance(N, int)
        if scale_norm:
            self.Norm = ScaleNorm(layer.size)
        else:
            self.Norm = LayerNorm(layer.size)
        self.N_layers = layer_clones(layer, N)

    def forward(self, x, mask, adj_matrix, imf_1_matrix, imf_2_matrix, imf_3_matrix, imf_4_matrix, 
                imf_5_matrix):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.N_layers:
            x = layer(x, mask, adj_matrix, imf_1_matrix, imf_2_matrix, imf_3_matrix, imf_4_matrix, 
                      imf_5_matrix)
        return self.Norm(x)


class Self_Attention_Block(nn.Module):
    """
    Self_Attention_Block is made up of self-attn block and feed forward block (defined below)
    """
    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm, use_adapter):
        super(Self_Attention_Block, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = layer_clones(SublayerConnection(size, dropout, scale_norm, use_adapter), 2)

    def forward(self, x, mask, adj_matrix, imf_1_matrix, imf_2_matrix, imf_3_matrix, imf_4_matrix, 
                imf_5_matrix):
        """Follow Figure 2 (left gray block) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, adj_matrix, imf_1_matrix, imf_2_matrix,
                                                         imf_3_matrix, imf_4_matrix, imf_5_matrix, 
                                                         mask)
                                                         )
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, scale_norm:bool, use_adapter:bool):
        super(SublayerConnection, self).__init__()
        if scale_norm:
            self.Norm = ScaleNorm(size)
        else:
            self.Norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.use_adapter = use_adapter
        if self.use_adapter:
            self.adapter = Adapter(size, 8)
        else:
            self.adapter = None

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        if self.use_adapter:
            return x + self.dropout(self.adapter(sublayer(self.Norm(x))))
        else:
            return x + self.dropout(sublayer(self.Norm(x)))



# =======================================================================================
### 3. Graph Representation Readout layer
class Graph_Representation_Readout(nn.Module):
    """
    Graph representation part
    """
    def __init__(self, n_nodes, d_model, aggregation_type='mean', n_output=1, n_layers=1, 
                 leaky_relu_slope=0.01, dropout=0.0, scale_norm=False):
        super(Graph_Representation_Readout, self).__init__()
        self.aggregation_type = aggregation_type
        self.d_model = d_model
        self.n_nodes = n_nodes
        # defined projected layer
        self.projected_layer = list()
        if n_layers == 1:
            self.projected_layer.append(nn.Linear(d_model, n_output))
        else:
            for _ in range(n_layers-1):
                self.projected_layer.append(nn.Linear(d_model, d_model))
                self.projected_layer.append(nn.LeakyReLU(leaky_relu_slope))
                self.projected_layer.append(ScaleNorm(d_model) if scale_norm else LayerNorm(d_model))
                self.projected_layer.append(nn.Dropout(dropout))
            self.projected_layer.append(nn.Linear(d_model, n_output))
        self.projected_layer = torch.nn.Sequential(*self.projected_layer)

    def _mean_aggregation(self, output_masked, mask, dim=1):
        """ mean """
        assert isinstance(dim, int) and dim != 0
        output_sum = output_masked.sum(dim=dim)
        mask_sum = mask.sum(dim=(dim))
        output_avg_pooling = output_sum / mask_sum
        # if mask_sum, we assume the out_sum is 0 and avg out pooling is 0
        return torch.nan_to_num(output_avg_pooling, nan=0)

    def _splitted_mean_aggregation(self, output_masked, mask, dim=1):
        """ splitted mean """
        assert isinstance(dim, int) and dim != 0
        output_sum = output_masked.sum(dim=dim)
        splitted_output_sum = torch.split(output_sum, int(self.d_model/self.n_nodes)+1, dim=dim)
        splitted_mask = torch.split(mask[:,:,0], 1, dim=dim)
        output_avg_pooling_tuple = ()
        for splitted_tensor, splitted_m in zip(splitted_output_sum, splitted_mask):
            avg_splitted_output_sum = splitted_tensor / splitted_m
            output_avg_pooling_tuple = output_avg_pooling_tuple + (avg_splitted_output_sum,)
        output_avg_pooling = torch.cat(output_avg_pooling_tuple, dim=dim)
        return torch.nan_to_num(output_avg_pooling, nan=0)

    def _sum_aggregation(self, output_masked, mask, dim=1):
        """ sum """
        assert isinstance(dim, int) and dim != 0
        out_sum = output_masked.sum(dim=dim)
        return out_sum

    def _dummy_node_aggregation(self, output_masked, mask, dim=1):
        """ dummy node """
        assert isinstance(dim, int) and dim != 0
        return output_masked[:,0]

    def _aggregation(self, aggregation_type, output_masked, mask):
        """
        Different type of aggregation, for example, mean, sum, dummy node, ... 
        Raise Error when unsupported aggregation type is found
        """
        if aggregation_type == 'mean':
            output_avg_pooling = self._mean_aggregation(output_masked, mask, dim=1)
        elif aggregation_type == 'splitted_mean':
            output_avg_pooling = self._splitted_mean_aggregation(output_masked, mask, dim=1)
        elif aggregation_type == 'sum':
            output_avg_pooling = self._sum_aggregation(output_masked, mask, dim=1)
        elif aggregation_type == 'dummy_node':
            output_avg_pooling = self._dummy_node_aggregation(output_masked, mask, dim=1)
        else:
            raise ValueError(f'Unsupported aggregation type: {aggregation_type}')
        return output_avg_pooling

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        output_masked = x * mask
        output_avg_pooling = self._aggregation(self.aggregation_type, output_masked, mask)
        projected = self.projected_layer(output_avg_pooling)
        return projected



# =======================================================================================
### Self Attention Block
class Time_Series_Multi_Head_Attention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, 
                 lambda_attention=DEFAULT_LAMBDAS_ATTENTION, 
                 lambda_imf_1=DEFAULT_LAMBDAS_IMF_1, 
                 lambda_imf_2=DEFAULT_LAMBDAS_IMF_2, 
                 lambda_imf_3=DEFAULT_LAMBDAS_IMF_3, 
                 lambda_imf_4=DEFAULT_LAMBDAS_IMF_4, 
                 lambda_imf_5=DEFAULT_LAMBDAS_IMF_5,
                 trainable_lambda=False, imf_matrix_kernel='softmax'
                 ):
        """Take in model size and number of heads."""
        super(Time_Series_Multi_Head_Attention, self).__init__()
        assert d_model % h == 0
        assert isinstance(trainable_lambda, bool)
        # Assumption: d_v == d_k
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.d_k = d_model // h
        self.h = h
        self.linears = layer_clones(nn.Linear(d_model, d_model), 4)
        self.trainable_lambda = trainable_lambda
        lambda_adjacency = 1. - lambda_attention - lambda_imf_1 - lambda_imf_2 - lambda_imf_3 - lambda_imf_4 - lambda_imf_5
        if trainable_lambda:
            lambdas_tensor = torch.tensor([lambda_attention, lambda_imf_1, lambda_imf_2, lambda_imf_3, lambda_imf_4, lambda_imf_5, lambda_adjacency], 
                                          requires_grad=True)
            self.lambdas = torch.nn.Parameter(lambdas_tensor)
        else:
            self.lambdas = (lambda_attention, lambda_imf_1, lambda_imf_2, lambda_imf_3, lambda_imf_4,
                            lambda_imf_5, lambda_adjacency)
        if imf_matrix_kernel == 'exp':
            self.imf_matrix_kernel = lambda x: torch.exp(-x)
        elif imf_matrix_kernel == 'softmax':
            self.imf_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
        elif imf_matrix_kernel == 'identical':
            self.imf_matrix_kernel = lambda x: x
        elif imf_matrix_kernel == 'absolute':
            self.imf_matrix_kernel = lambda x: torch.abs(x)
        else:
            raise ValueError(f"Unknown imf matrix kernel function: {imf_matrix_kernel}")

    def _attention(self, Query, Key, Value, p_adj, p_imf_1, p_imf_2, p_imf_3, p_imf_4, p_imf_5,
                    mask=None, dropout=None, 
                    lambdas=DEFAULT_LAMBDAS, trainable_lambda=False,
                    inf=1e12,
                    ):
        """
        Self-attention operation for equation 9 and feature enhancement part.
        """
        d_k = Query.size(-1)
        scores = torch.matmul(Query, Key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).repeat(1, Query.shape[1], Query.shape[2], 1) == 0, -inf)
        p_attn = F.softmax(scores, dim = -1)

        if trainable_lambda:
            P_mat = [p_attn, p_imf_1, p_imf_2, p_imf_3, p_imf_4, p_imf_5, p_adj]
            p_weighted = sum([u*v for (u, v) in zip(lambdas, P_mat)])
        
        if dropout is not None:
            p_weighted = dropout(p_weighted)

        x = torch.matmul(p_weighted, Value)
        return x, p_weighted, p_attn

    def _preprocess_imf_matrices(self, 
                                 imf_1_mat, imf_2_mat, imf_3_mat, imf_4_mat, imf_5_mat, 
                                 Query, mask, imf_matrix_kernel):
        list_of_imf_matrices = [imf_1_mat, imf_2_mat, imf_3_mat, imf_4_mat, imf_5_mat]
        res = list()
        for imf_matrix in list_of_imf_matrices:
            imf_matrix = imf_matrix.masked_fill(mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
            imf_matrix = imf_matrix_kernel(imf_matrix)
            res.append(imf_matrix.unsqueeze(1).repeat(1, Query.shape[1], 1, 1))
        return res

    def _linear_projections(self, n_batches, Query, Key, Value):
        return [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (Query, Key, Value))]

    def _preprocess_adj_matrix(self, adj_matrix, Query, eps=1e-6):
        adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
        adj_matrix = adj_matrix.unsqueeze(1).repeat(1, Query.shape[1], 1, 1)
        return adj_matrix

    def _concat_output_matrix(self, x, n_batches):
        return x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)

    def forward(self, Query, Key, Value, adj_matrix, imf_1_matrix, imf_2_matrix, imf_3_matrix,
                imf_4_matrix, imf_5_matrix, mask=None):
        "Implements Equation 9"
        if mask is not None:
            mask = mask.unsqueeze(1)    # Same mask applied to all h heads.
        n_batches = Query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        Query, Key, Value = self._linear_projections(n_batches, Query, Key, Value)

        # Prepare the imf matrices
        p_imf_1, p_imf_2, p_imf_3, p_imf_4, p_imf_5 = self._preprocess_imf_matrices(
            imf_1_matrix, imf_2_matrix, imf_3_matrix, imf_4_matrix, imf_5_matrix,
            Query=Query, mask=mask, imf_matrix_kernel=self.imf_matrix_kernel
            )

        # Prepare the adj_matrix
        p_adj = self._preprocess_adj_matrix(adj_matrix, Query=Query)
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn, self.self_attn = self._attention(Query, Key, Value, p_adj, 
                                                        p_imf_1, p_imf_2, p_imf_3, p_imf_4, p_imf_5,
                                                        mask=mask, dropout=self.dropout,
                                                        lambdas=self.lambdas,
                                                        trainable_lambda=self.trainable_lambda,
                                                        )
        
        # 3) "Concat" using a view and apply a final linear.
        x = self._concat_output_matrix(x, n_batches=n_batches)
        return self.linears[-1](x)


### Sequential feed forward
class Sequential_FFN(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.0, 
                 dense_output_nonlinearity='relu'):
        super(Sequential_FFN, self).__init__()
        self.N_dense = N_dense
        self.linears = layer_clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = layer_clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x

    def forward(self, x):
        if self.N_dense == 0:
            return x
        
        for i in range(len(self.linears)-1):
            x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))
            
        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))



# =======================================================================================
### Norm
class ScaleNorm(nn.Module):
    """Scale Norm"""
    "All g’s in SCALE NORM are initialized to sqrt(d)"
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps
    
    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


### Norm
class LayerNorm(nn.Module):
    """Layer Norm"""
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a, self.b = nn.Parameter(torch.ones(features)), nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mu, sigma = x.mean(-1, keepdim=True), x.std(-1, keepdim=True)
        return self.a * (x - mu) / (sigma + self.eps) + self.b


### layer clones tool
def layer_clones(module, N):
    """
    Stack N identical layers.

    :return: layers
    """
    assert isinstance(N, int) and N != 0
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

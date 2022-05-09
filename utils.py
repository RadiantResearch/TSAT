import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_
import warnings



class TSAT_parameter():
    """
    The TSAT's parameter is defined here. It contains model parameter and training parameter. Different dataset use different parameter setting.

    :param dataset_name
    """
    def __init__(self, dataset_name:str):
        self.model_parameters, self.training_parameters = None, None
        assert dataset_name != None     # The name of dataset should not be None, please check dataset_name input correctly
        if dataset_name == 'temp':
            self.model_parameters = {
            'l_backcast': None,                             # backcast length
            'd_edge': None,                                 # edge features (number of IMF used)
            'd_model': None,                                # model hidden layer
            'N': None,                                      # number of Self_Attention_Block
            'h': None,                                      # Multi-attention heads
            'N_dense': None,                                # Sequential feed forward layers
            'n_output': None,
            'n_nodes':None,
            'lambda_attention': None,
            'lambda_imf_1': None,
            'lambda_imf_2': None,
            'lambda_imf_3': None,
            'lambda_imf_4': None,
            'lambda_imf_5': None,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # =====================================================================================
        # ======================================= ETTh1 =======================================
        # ETTh1_24
        elif dataset_name == 'ETTh1_24':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 24*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTh1_48
        elif dataset_name == 'ETTh1_48':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 48*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTh1_168
        elif dataset_name == 'ETTh1_168':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 168*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTh1_336
        elif dataset_name == 'ETTh1_336':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 336*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTh1_336
        elif dataset_name == 'ETTh1_720':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 720*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # =====================================================================================
        # ======================================= ETTh2 =======================================
        # ETTh2_24
        elif dataset_name == 'ETTh2_24':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 24*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTh2_48
        elif dataset_name == 'ETTh2_48':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 48*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTh2_168
        elif dataset_name == 'ETTh2_168':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 168*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTh2_336
        elif dataset_name == 'ETTh2_336':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 336*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTh2_336
        elif dataset_name == 'ETTh2_720':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 720*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # =====================================================================================
        # ======================================= ETTm1 =======================================
        # ETTm1_24
        elif dataset_name == 'ETTm1_24':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 24*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTm1_48
        elif dataset_name == 'ETTm1_48':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 48*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': 'relu',
            'imf_matrix_kernel': 'exp',
            'dropout': 0.0,
            'aggregation_type': 'mean',
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTm1_168
        elif dataset_name == 'ETTm1_168':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 168*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTm1_336
        elif dataset_name == 'ETTm1_336':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 336*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # ETTm1_336
        elif dataset_name == 'ETTm1_720':    #
            self.model_parameters = {
            'l_backcast': 720,                          # backcast length
            'd_edge': 5,                           # edge features (number of IMF used)
            'd_model': 256,                         # model hidden layer
            'N': 8,                                 # number of Self_Attention_Block
            'h': 16,                                 # Multi-attention heads
            'N_dense': 1,                           # Sequential feed forward layers
            'n_output': 720*7,
            'n_nodes':7,
            'lambda_attention': 0.33,
            'lambda_imf_1': 0.0999,
            'lambda_imf_2': 0.0801,
            'lambda_imf_3': 0.06,
            'lambda_imf_4': 0.0399,
            'lambda_imf_5': 0.0201,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # =====================================================================================
        # ======================================== WTH ========================================
        # WTH_48
        elif dataset_name == 'WTH_48':
            self.model_parameters = {
                'l_backcast': 720,                          # backcast length
                'd_edge': 5,                           # edge features (number of IMF used)
                'd_model': 256,                         # model hidden layer
                'N': 8,                                 # number of Self_Attention_Block
                'h': 16,                                 # Multi-attention heads
                'N_dense': 1,                           # Sequential feed forward layers
                'n_output': 48*9,
                'n_nodes':9,
                'lambda_attention': 0.33,
                'lambda_imf_1': 0.0999,
                'lambda_imf_2': 0.0801,
                'lambda_imf_3': 0.06,
                'lambda_imf_4': 0.0399,
                'lambda_imf_5': 0.0201,
                'dense_output_nonlinearity': 'relu',
                'imf_matrix_kernel': 'exp',
                'dropout': 0.0,
                'aggregation_type': 'mean',
                'scale_norm': True,
                'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # WTH_168
        elif dataset_name == 'WTH_168':
            self.model_parameters = {
                'l_backcast': 720,                          # backcast length
                'd_edge': 5,                           # edge features (number of IMF used)
                'd_model': 256,                         # model hidden layer
                'N': 8,                                 # number of Self_Attention_Block
                'h': 16,                                 # Multi-attention heads
                'N_dense': 1,                           # Sequential feed forward layers
                'n_output': 48*9,
                'n_nodes':9,
                'lambda_attention': 0.33,
                'lambda_imf_1': 0.0999,
                'lambda_imf_2': 0.0801,
                'lambda_imf_3': 0.06,
                'lambda_imf_4': 0.0399,
                'lambda_imf_5': 0.0201,
                'dense_output_nonlinearity': 'relu',
                'imf_matrix_kernel': 'exp',
                'dropout': 0.0,
                'aggregation_type': 'mean',
                'scale_norm': True,
                'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # WTH_336
        elif dataset_name == 'WTH_336':
            self.model_parameters = {
                'l_backcast': 720,                          # backcast length
                'd_edge': 5,                           # edge features (number of IMF used)
                'd_model': 256,                         # model hidden layer
                'N': 8,                                 # number of Self_Attention_Block
                'h': 16,                                 # Multi-attention heads
                'N_dense': 1,                           # Sequential feed forward layers
                'n_output': 336*9,
                'n_nodes':9,
                'lambda_attention': 0.33,
                'lambda_imf_1': 0.0999,
                'lambda_imf_2': 0.0801,
                'lambda_imf_3': 0.06,
                'lambda_imf_4': 0.0399,
                'lambda_imf_5': 0.0201,
                'dense_output_nonlinearity': 'relu',
                'imf_matrix_kernel': 'exp',
                'dropout': 0.0,
                'aggregation_type': 'mean',
                'scale_norm': True,
                'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # WTH_720
        elif dataset_name == 'WTH_720':
            self.model_parameters = {
                'l_backcast': 720,                          # backcast length
                'd_edge': 5,                           # edge features (number of IMF used)
                'd_model': 256,                         # model hidden layer
                'N': 8,                                 # number of Self_Attention_Block
                'h': 16,                                 # Multi-attention heads
                'N_dense': 1,                           # Sequential feed forward layers
                'n_output': 720*9,
                'n_nodes':9,
                'lambda_attention': 0.33,
                'lambda_imf_1': 0.0999,
                'lambda_imf_2': 0.0801,
                'lambda_imf_3': 0.06,
                'lambda_imf_4': 0.0399,
                'lambda_imf_5': 0.0201,
                'dense_output_nonlinearity': 'relu',
                'imf_matrix_kernel': 'exp',
                'dropout': 0.0,
                'aggregation_type': 'mean',
                'scale_norm': True,
                'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }


    def parameters(self):
        """
        Return the model parameter and training parameter
        """
        return self.model_parameters, self.training_parameters

    # def creat_dataset_parameter(self, mp, tp):
    #     self.model_parameters = mp
    #     self.training_parameters = tp
    #     return self



def xavier_normal_small_init_(tensor, gain=1.):
    """
    Type: (Tensor, float) -> Tensor
    :param tensor

    :return: tensor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    return _no_grad_normal_(tensor, 0., std)


def xavier_uniform_small_init_(tensor, gain=1.):
    """
    Type: (Tensor, float) -> Tensor
    :param tensor

    :return: tensor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return _no_grad_uniform_(tensor, -a, a)


def loss_function(loss_function_name:str):
    """
    Define the loss function here
    :param loss_function_name: the name of loss function

    :return: a PyTorch loss function
    """
    assert loss_function_name != None   # the loss function should not be none
    if loss_function_name == 'rmse':
        return torch.nn.MSELoss()
    elif loss_function_name == 'mae':
        return torch.nn.L1Loss()
    elif loss_function_name == 'smoothed mae':
        return torch.nn.SmooothL1Loss()
    elif loss_function_name == 'Cross Entropy Loss':
        return torch.nn.CrossEntropyLoss
    elif loss_function_name == 'Huber Loss':
        return torch.nn.HuberLoss()
    elif loss_function_name == 'bce':
        return torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        assert f'Unsupported loss function name : {loss_function_name}'

def calculate_loss(y_true, y_pred, loss_function_name, criterion, device):
    """
    y_true.shape = (batch, num_tasks)
    y_pred.shape = (batch, num_tasks)
    """
    if loss_function_name == 'mae':
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        loss = criterion(y_pred, y_true)
    elif loss_function_name == 'smoothed mae':
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        loss = criterion(y_pred, y_true)
    elif loss_function_name == 'rmse':
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        loss = torch.sqrt(criterion(y_pred, y_true))
    elif loss_function_name == 'Huber Loss':
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        loss = criterion(y_pred, y_true)
    elif loss_function_name == 'bce':
        # find all -1 in y_true
        y_true = y_true.long()
        y_mask = torch.where(y_true == -1, torch.tensor([0]).to(device), torch.tensor([1]).to(device))
        y_cal_true = torch.where(y_true == -1, torch.tensor([0]).to(device), y_true).float()
        loss = criterion(y_pred, y_cal_true) * y_mask
        loss = loss.sum() / y_mask.sum()
    else:
        loss = criterion(y_pred, y_true)
    return loss

def evaluation(y_true, y_pred, y_graph_name, requirement, data_mean=0, data_std=1):
    """
    This function is to evaluate the result of y_true and y_pred and calculate the corresponding loss measurement
    
    y_true.shape = (samples, task_numbers)
    y_pred.shape = (samples, task_numbers)

    :return: collect_result: a dict consists of different items, e.g. rmse, sample
    """
    collect_result = {}
    assert len(requirement) != 0    # the requirement should not be empty
    if 'sample' in requirement:
        collect_result['graph_name'] = y_graph_name.tolist()
        collect_result['prediction'] = (y_pred * data_std + data_mean).tolist()
        collect_result['label'] = y_true.tolist()
    if 'rmse' in requirement:
        # y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        collect_result['rmse'] = np.sqrt(F.mse_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean'))
    if 'mse and mae' in requirement:
        # y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        collect_result['mse'] = F.mse_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean')
        collect_result['mae'] = F.l1_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean')
    if 'mae' in requirement:
        y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        collect_result['mae'] = F.l1_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean')
    if 'auc' in requirement:
        auc_score_list = []
        if y_true.shape[1] > 1:
            for label in range(y_true.shape[1]):
                true, pred = y_true[:, label], y_pred[:, label]
                # all 0's or all 1's
                if len(set(true)) == 1:
                    auc_score_list.append(float('nan'))
                else:
                    auc_score_list.append(metrics.roc_auc_score(true[np.where(true >= 0)], pred[np.where(true >= 0)]))
            collect_result['auc'] = np.nanmean(auc_score_list)
        else:
            collect_result['auc'] = metrics.roc_auc_score(y_true, y_pred)
    if 'bce' in requirement:
        # find all -1 in y_true
        y_mask = np.where(y_true == -1, 0, 1)
        y_cal_true = np.where(y_true == -1, 0, y_true)
        loss = F.binary_cross_entropy_with_logits(torch.tensor(y_pred), torch.tensor(y_cal_true), reduction='none') * y_mask
        collect_result['bce'] = loss.sum() / y_mask.sum()
    # Checking any missing requirement that not yet cover
    for item in requirement:
        if item not in ['sample', 'rmse', 'mse and mae', 'mae', 'auc', 'bce']:
            warnings.warn(f'{item} is not in requirement. Therefore, the collect result does not contain this.')
    return collect_result

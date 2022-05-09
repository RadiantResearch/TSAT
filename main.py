import argparse
from collections import defaultdict
from dataset_TSAT_ETTm1_48 import generate_ETTm1_data
from dataset_graph import construct_TSAT_dataset, graph_collate_func_TSAT_normalization_require
import numpy as np
from sklearn.model_selection import train_test_split
from TSAT import make_TSAT_model
import time
import torch
from torch.utils.data import DataLoader
from utils import TSAT_parameter, loss_function, calculate_loss



class KOI_model_train_test_interface():
    def __init__(self, TSAT_model, model_params:dict, train_params:dict) -> None:
        self.TSAT_model = TSAT_model
        self.TSAT_model = self.TSAT_model.to(train_params['device'])    # send the model to GPU
        self.train_params = train_params
        self.model_params = model_params
        self.criterion = loss_function(train_params['loss_function'])
        # self.optimizer = torch.optim.Adam(self.TSAT_model.parameters())

    def import_dataset(self, dataset) -> None:
        # import the dataset
        train_valid_split_ratio = 0.2
        num_workers = 20 # 4
        train_valid_dataset, self.test_dataset = train_test_split(dataset, test_size=train_valid_split_ratio, shuffle=True)  # False
        self.train_dataset, self.valid_dataset = train_test_split(train_valid_dataset, test_size=train_valid_split_ratio)
        # Data Loader
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.train_params['batch_size'], 
                                       collate_fn=graph_collate_func_TSAT_normalization_require, shuffle=True,
                                       drop_last=True, num_workers=num_workers, pin_memory=False)
        
        self.valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.train_params['batch_size'], 
                                       collate_fn=graph_collate_func_TSAT_normalization_require, shuffle=True, 
                                       drop_last=True, num_workers=num_workers, pin_memory=False)

    def calculate_number_of_parameter(self) -> int:
        model_parameters = filter(lambda p: p.requires_grad, self.TSAT_model.parameters())
        return int(sum([np.prod(p.size()) for p in model_parameters]))

    def view_train_params(self):
        # view the training parameters
        return self.train_params
    
    def view_model_params(self):
        # view the model parameters
        return self.model_params

    def train_model(self) -> None:
        # training start
        start_time = time.time()
        self.TSAT_model.train()
        for batch in self.train_loader:
            graph_name_list, adjacency_matrix, node_features, imf_1_matrices, imf_2_matrices, imf_3_matrices, imf_4_matrices, imf_5_matrices, y_true_normalization, _, _ = batch
            adjacency_matrix = adjacency_matrix.to(self.train_params['device'])  # (batch, max_length, max_length)
            node_features = node_features.to(self.train_params['device'])  # (batch, max_length, d_node)
            imf_1_matrices = imf_1_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_2_matrices = imf_2_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_3_matrices = imf_3_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_4_matrices = imf_4_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_5_matrices = imf_5_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            y_true_normalization = y_true_normalization.to(self.train_params['device'])  # (batch, task_numbers)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
            y_pred_normalization = self.TSAT_model(
                node_features, batch_mask, adjacency_matrix, imf_1_matrices, imf_2_matrices, imf_3_matrices, imf_4_matrices, imf_5_matrices
                )
            loss = calculate_loss(y_true_normalization, y_pred_normalization, self.train_params['loss_function'], self.criterion, self.train_params['device'])
        # save model
        end_time = time.time()
        print(f'TSAT train complete! Training time: {end_time-start_time}')
    
    def test_model(self) -> None:
        # testing start
        start_time = time.time()
        # load model and no_grad
        end_time = time.time()
        print(f'TSAT test complete! Testing time: {end_time-start_time}')



if __name__ == '__main__':
    ## init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help='gpu', default=0)
    parser.add_argument("--dataset", type=str, help='name of dataset', default='ETTm1_48')
    args = parser.parse_args()
    print(args)

    TSAT_parameters = TSAT_parameter(args.dataset)
    model_params, train_params = TSAT_parameters.parameters()

    ## Check GPU is available
    train_params['device'] = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    # generate graph from data
    data_graph, data_label = generate_ETTm1_data(
        n_lookback_days=720, 
        n_lookforward_days=48, 
        adj_mat_method='correlation', 
        use_tqdm=True
        )
    # data_graph, data_label = generate_ETTm1_data(
    #     n_lookback_days=720, 
    #     n_lookforward_days=48, 
    #     adj_mat_method='zero_mat', 
    #     use_tqdm=True
    #     )

    dataset = construct_TSAT_dataset(data_graph, data_label, normalization_require=True)
    total_metrics = defaultdict(list)

    ## main train and test
    TSAT_model = make_TSAT_model(**model_params)
    model_interface = KOI_model_train_test_interface(TSAT_model, model_params, train_params)
    model_interface.import_dataset(dataset=dataset)
    model_interface.train_model()
    model_interface.test_model()

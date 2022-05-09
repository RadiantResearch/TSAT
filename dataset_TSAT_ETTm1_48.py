from ast import parse
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

from PyEMD import EMD


def emd_imf(signal):
    """
    This function is to calculate EMD of the time series.
    :params: signal: list

    :return: res_dict: a dict consists of the different imf list value
    """
    if isinstance(signal, list):
        signal = np.array(signal)
    assert isinstance(signal, np.ndarray)
    IMFs = EMD().emd(signal, np.arange(len(signal)))
    res_dict = {}
    for _ in range(IMFs.shape[0]):
        res_dict[f'imf_{_}'] = IMFs[_].tolist()
    return res_dict


def calculate_imf_features(n_zones, index_pair_for_one, zones_dict:dict, ticker_SMD_dict:dict, n_imf_use=5) -> np.ndarray:
    """
    compute the EMD.

    :return: imf_features
    """
    assert isinstance(n_imf_use, int)
    imf_features = np.zeros((n_zones, n_zones, n_imf_use))
    ticker_A, ticker_B = None, None
    for pair in index_pair_for_one:
        if ticker_A != zones_dict[pair[0]]:
            ticker_A = zones_dict[pair[0]]
            ticker_A_SMD = ticker_SMD_dict[ticker_A]
        if ticker_B != zones_dict[pair[1]]:
            ticker_B = zones_dict[pair[1]]
            ticker_B_SMD = ticker_SMD_dict[ticker_B]
        
        ef = [0] * n_imf_use
        for n_imf in list(range(1, n_imf_use+1)):  # n_imf_to_exact = n_imf_use
            if f'imf_{n_imf}' in ticker_A_SMD and f'imf_{n_imf}' in ticker_B_SMD:
                # to get both imf for both 2 tickers
                ef[n_imf-1] = (np.corrcoef(ticker_A_SMD[f'imf_{n_imf}'],
                                            ticker_B_SMD[f'imf_{n_imf}'])[0][1]
                                )
            else:  # exit the loop when there is no further imf correctlation
                break
        imf_features[pair[0]][pair[1]], imf_features[pair[1]][pair[0]] = np.array(ef), np.array(ef)
    
    return imf_features


def process_data(df, n_lookback_days, n_lookforward_days, adj_mat_method='fully_connected', use_tqdm=True, **kwargs):
    """
    This is the main part of generate graph.
    """
    if n_lookforward_days > n_lookback_days:
        warnings.warn(f'The number of lookforward days ({n_lookforward_days}) is lager than lookback days ({n_lookback_days}). Please conside using longer lookback days')
    _graph_data, _graph_label = [], []
    zones = df.columns.to_list()
    n_zones, zones_dict = len(zones), dict(zip(range(len(zones)), zones))
    # nf_global_max, nf_global_min = df.max().to_numpy(), df.min().to_numpy()
    ranges = range(len(df) - n_lookforward_days - n_lookback_days)
    print(f'tqdm used: {use_tqdm}')
    range_iterable = tqdm(ranges) if use_tqdm else ranges
    for adate in range_iterable:
        if not use_tqdm:
            print(f'Generating graph for time {df.index[adate+n_lookback_days]} ...... ', end='')
        
        lookback_period_df = df[adate:adate+n_lookback_days]
        lookforward_period_df = df[adate+n_lookback_days:adate+n_lookback_days+n_lookforward_days]
    
        # node_features
        node_features = lookback_period_df.to_numpy().transpose()
        
        # node_features (normalization)
        nf_max, nf_min = np.amax(node_features, axis=1), np.amin(node_features, axis=1)
        # if encounter constant series, i.e., max = min. The result will be all zero.
        if np.any(nf_max == nf_min):
            nf_max[np.where(nf_max - nf_min == 0)], nf_min[np.where(nf_max - nf_min == 0)] = 1, 0
        
        nf_MIN = np.repeat(nf_min, n_lookback_days, axis=0).reshape(n_zones, n_lookback_days)
        nf_MAX = np.repeat(nf_max, n_lookback_days, axis=0).reshape(n_zones, n_lookback_days)
        node_features_normalization = (node_features-nf_MIN)/(nf_MAX - nf_MIN)
        
        # adj_matrix (fully connected or corr)
        if adj_mat_method == 'fully_connected':
            # all 1 except the diagonal
            adj_mat = np.ones((n_zones, n_zones)) - np.eye(n_zones)
        elif adj_mat_method == 'correlation':
            # based on correlation
            correlation_matrix = np.abs(np.corrcoef(lookback_period_df.values, rowvar=False))
            correlation_matrix = np.where(correlation_matrix == 1., 0, correlation_matrix)
            correlation_matrix = np.where(correlation_matrix >= 0.8, 1, 0) # 0.75
            adj_mat = correlation_matrix
        elif adj_mat_method == 'zero_mat':
            # zero matrix
            adj_mat = np.zeros((n_zones, n_zones))
        elif adj_mat_method == 'random':
            # random
            b = np.random.random_integers(0, 1, size=(n_zones, n_zones))
            adj_mat = b * b.T
        else:
            raise TypeError(f'Unsupported adj_matrix method: {adj_mat_method}!')
        
        ## calculate imf_features
        index_pair_for_one = np.argwhere(np.triu(adj_mat) == 1)   # get the index pair form upper triangle part of adj_mat
        ticker_SMD_dict = dict.fromkeys(zones)
        involde_index_idxs_np = np.unique(index_pair_for_one.flatten())
        for index_idx in involde_index_idxs_np:
            ticker = zones_dict[index_idx]
            ticker_SMD_dict[ticker] = emd_imf(lookback_period_df[ticker].to_list())
        
        imf_features = calculate_imf_features(n_zones, index_pair_for_one, zones_dict, ticker_SMD_dict, n_imf_use=5)

        # split the imd features into dict
        imf_matries_dict = {
            'imf_1_matix':imf_features[:,:,0],
            'imf_2_matix':imf_features[:,:,1],
            'imf_3_matix':imf_features[:,:,2],
            'imf_4_matix':imf_features[:,:,3],
            'imf_5_matix':imf_features[:,:,4],
        }
        
        # label
        label = lookforward_period_df.to_numpy().transpose()
        
        # label (normalization)
        label_MIN = np.repeat(nf_min, n_lookforward_days, axis=0).reshape(n_zones, n_lookforward_days)
        label_MAX = np.repeat(nf_max, n_lookforward_days, axis=0).reshape(n_zones, n_lookforward_days)
        label_normalization = (label-label_MIN)/(label_MAX - label_MIN)
        label_normalization = label_normalization.flatten()
        
        name_of_sample = f"ETTm1_LookbackInculde[{lookback_period_df.index[0]} to {lookback_period_df.index[-1]}]_Predict[{lookforward_period_df.index[0]} to {lookforward_period_df.index[-1]}]"
        
        
        G = {
            'date': name_of_sample,
            'node_feat': node_features_normalization,
            'imf_matries_dict': imf_matries_dict,
            'adj_mat': adj_mat,
            'node_local_MAX': nf_max,
            'node_local_MIN': nf_min,
            'node_global_MAX': None,
            'node_global_MIN': None
        }
        _graph_data.append(G)
        _graph_label.append(list(label_normalization))
        if not use_tqdm:
            print('Done !')
            
    return _graph_data, _graph_label


def list_split(L, n_split):
    """
    Split the list
    """
    assert isinstance(L, list)
    k, m = divmod(len(L), n_split)
    _ = (L[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_split))
    return list(_)


def generate_ETTm1_data(n_lookback_days:int, n_lookforward_days:int, adj_mat_method:str, use_tqdm:bool):
    import time
    start_time = time.time()
    df = pd.read_csv(f'Data/ETTm1_{n_lookforward_days}/source/ETTm1.csv', index_col='date')
    assert len(df) != 0 # check the input df, it should noe be None.
    df = df[:16700]
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    graph_data, graph_label = process_data(df, n_lookback_days, n_lookforward_days, adj_mat_method=adj_mat_method, use_tqdm=use_tqdm)    # correlation
    assert len(graph_data) == len(graph_label)  # the length of data_mol and label should be same
    assert graph_data[0] is not None      # Check the datamol should be non-empty
    end_time = time.time()
    print(f'Generate ETTm1 data complete! Total time: {end_time-start_time}')
    return graph_data, graph_label

# ======================================================================================


if __name__ == '__main__':
    # init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--lb", type=int, help="n_lookback_days", default=720)
    parser.add_argument("--lf", type=int, help="n_lookforward_days", default=48)
    parser.add_argument("--adj_mat_method", type=str, help="adj_mat_method", default='zero_mat', choices=['zero_mat', 'fully_connected', 'correlation', 'random'])
    parser.add_argument("--use_tqdm", type=str, help="use tqdm or not", default='True', choices=['True', 'False'])
    args = parser.parse_args()
    assert isinstance(args.lb, int)
    assert isinstance(args.lf, int)
    # generate ETTm1_48 datas
    graph_data, graph_label = generate_ETTm1_data(args.lb, args.lf, args.adj_mat_method, bool(args.use_tqdm))

# from Public_version.dataset_TSAT_ETTm1_48 import generate_ETTm1_data

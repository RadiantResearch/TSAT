# Time Series Attention Transformer (TSAT)
![Python 3.8](https://img.shields.io/badge/Python-3.8-green.svg?style=plastic)
![PyTorch 1.8](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

The official implementation of the Time Series Attention Transformer (TSAT).

<p align='center'>
<image src="src/TSAT.png" alt="architecture" width="600"/>
</p>

## Code
- `main.py` : main TSAT model interface with training and testing
- `TSAT.py` : with TSAT class implementation
- `utils.py` : utility functions and dataset parameter
- `dataset_TSAT_ETTm1_48.py` : generate graph from dataset ETTm1

## Data Preparation

### ETT dataset
Download the Electricity Transformer Temperature Dataset from https://github.com/zhouhaoyi/ETDataset. Uncompress them and move the .csv to the `Data` folder.

### Multivariate Time series Data sets
The Electricity consumption dataset can be found on https://github.com/laiguokun/multivariate-time-series-data.

## Model parameters
The parameters setting can be found in `utils.py`.
- `l_backcast` : lengths of backcast
- `d_edge` : number of IMF used
- `d_model` : the time embedding dimension
- `N` : number of Self_Attention_Block
- `h` : number of head in Multi-head-attention
- `N_dense` : number of linear layer in Sequential feed forward layers
- `n_output` : number of output (lengths of forecast $\times$ number of node)
- `n_nodes` : number of node (aka number of time series)
- `lambda` : the initial value of the trainable lambda $\alpha_i$
- `dense_output_nonlinearity` the nonlinearity function in dense output layer

## Requirements
- Python 3.8
- PyTorch = 1.8.0 (with corresponding CUDA version)
- Pandas = 1.4.0
- Numpy = 1.22.2
- PyEMD = 1.2.1

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Contact
If you have any questions, please feel free to contact William Ng (Email: william.ng@koiinvestments.com).

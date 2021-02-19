import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')
from time import time
import random

from model import ngcf
from model.ngcf import NGCF
from util.loaddata import Data
import multiprocessing
import heapq
import util.metrics as metrics

def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)

def set_device():
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    return device


if __name__ == '__main__':

    seed = 1024
    set_seed(seed)

    device = set_device()

    filepath = 'amazon-book'
    data_generator = Data(path=filepath)
    plain_adj, norm_adj, mean_adj = data_generator.create_adj_mat()
    ngcf.data_generator = data_generator   # batch_size=args.batch_size

    model = NGCF(data_generator.n_users, data_generator.n_items, norm_adj,
                embedding_size = 10, l2_reg_embedding=0.00001, gcn_layers=3, drop_rate=0.3,
                device=device)
    model.fit(learning_rate=0.001, batch_size=2000, epochs=50, verbose=5, early_stop=False)


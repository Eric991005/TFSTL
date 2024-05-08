import logging
import numpy as np
import os
import pickle
import sys
import torch
from torch.utils.data import Dataset
import math
# from pytorch_wavelets import DWT1DForward, DWT1DInverse
from statsmodels.tsa.seasonal import STL
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import csv

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        
    return mae, rmse, mape





def _compute_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def stl_decomposition(series):
    """
    Perform STL decomposition on each column of an ndarray, returning the original series, trend, and seasonal components.
    
    Args:
    series: ndarray, shape (n_samples, n_features)
    
    Returns:
    original_series: ndarray, shape (n_samples, n_features)
    trend_series: ndarray, shape (n_samples, n_features)
    seasonal_series: ndarray, shape (n_samples, n_features)
    """
    trend_series = np.zeros(series.shape)
    seasonal_series = np.zeros(series.shape)

    # Simply return the original series
    original_series = series

    for i in range(series.shape[1]):
        # Perform STL decomposition on each column
        stl = STL(series[:, i], period=10, robust=False)
        res = stl.fit()

        # Retrieve the trend and seasonal components
        trend_series[:, i] = res.trend
        seasonal_series[:, i] = res.seasonal

    return original_series, trend_series, seasonal_series


class TradeDataset(Dataset):
    def __init__(self, args, mode, mean=None, std=None):
        self.args = args
        self.mode = mode
        self.trade = np.load(args.trade_file)['result']
        self.num_step = self.trade.shape[0]

        if mode in ['train', 'val', 'test']:
            self.train_steps = round(args.train_ratio * self.num_step)
            self.test_steps = round(args.test_ratio * self.num_step)
            self.val_steps = self.num_step - self.train_steps - self.test_steps

            if mode == 'train':
                self.data = self.trade[:self.train_steps]
            elif mode == 'val':
                self.data = self.trade[self.train_steps:self.train_steps + self.val_steps]
            elif mode == 'test':
                self.data = self.trade[-self.test_steps:]
        elif mode == 'whole':
            self.data = self.trade  # Using the entire dataset
        
        self.original, self.trend, self.seasonal = stl_decomposition(self.data)
        self.X, self.Y = seq2instance(self.data, args.T1, args.T2)
        self.XL, self.YL = seq2instance(self.trend, args.T1, args.T2)
        self.XH, self.YH = seq2instance(self.seasonal, args.T1, args.T2)

        if mode == 'train':
            self.mean, self.std = np.mean(self.X), np.std(self.X)
        elif mode in ['val', 'test', 'whole']:
            if mean is None or std is None:
                # If mean and std are not provided in the non-training mode, they are computed from the data of the current mode
                self.mean, self.std = np.mean(self.X), np.std(self.X)
            else:
                self.mean, self.std = mean, std

        self.XL, self.XH = (self.XL - self.mean) / self.std, (self.XH - self.mean) / self.std
        self.X = (self.X - self.mean) / self.std

        # Compute temporal embedding
        self.TE = self.compute_te(args, self.num_step)

        if mode in ['train', 'val', 'test']:
            if mode == 'train':
                self.te_data = self.TE[:self.train_steps]
            elif mode == 'val':
                self.te_data = self.TE[self.train_steps:self.train_steps + self.val_steps]
            elif mode == 'test':
                self.te_data = self.TE[-self.test_steps:]
        elif mode == 'whole':
            self.te_data = self.TE  # Using whole time embedding in 'whole' mode

        self.TE = seq2instance(self.te_data, args.T1, args.T2)
        self.TE = np.concatenate(self.TE, axis=1).astype(np.int32)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = (torch.tensor(self.X[index]), torch.tensor(self.XL[index]), torch.tensor(self.XH[index]), torch.tensor(self.TE[index]))
        y = (torch.tensor(self.Y[index]), torch.tensor(self.YL[index]))
        return x, y

    def compute_te(self, args, num_step):
        # Add your implementation
        TE = np.zeros([num_step, 2])
        startd = (1 - 1) * 3
        df = 4
        startt = 0
        for i in range(num_step):
            TE[i,0] = startd //  3
            startd = (startd + 1) % (df * 3)
            TE[i,1] = startt
            startt = (startt + 1) % 3
        return TE
    
def save_to_csv(file_path, data):
    # Check if the directory exists, if not, create it
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Write data to CSV
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)
    print(f"Data saved to {file_path}")  # You might want to replace this with your logging method
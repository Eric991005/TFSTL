import logging
import numpy as np
import os
import pickle
import sys
import torch
import math
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from statsmodels.tsa.seasonal import STL
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import csv
# from vmdpy import VMD

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

def disentangle(data, w, j):
    # Disentangle
    dwt = DWT1DForward(wave=w, J=j)
    idwt = DWT1DInverse(wave=w)
    torch_traffic = torch.from_numpy(data).transpose(1,-1).reshape(data.shape[0]*data.shape[2], -1).unsqueeze(1)
    torch_trafficl, torch_traffich = dwt(torch_traffic.float())
    placeholderh = torch.zeros(torch_trafficl.shape)
    placeholderl = []
    for i in range(j):
        placeholderl.append(torch.zeros(torch_traffich[i].shape))
    torch_trafficl = idwt((torch_trafficl, placeholderl)).reshape(data.shape[0],data.shape[2],1,-1).squeeze(2).transpose(1,2)
    torch_traffich = idwt((placeholderh, torch_traffich)).reshape(data.shape[0],data.shape[2],1,-1).squeeze(2).transpose(1,2)
    trafficl = torch_trafficl.numpy()
    traffich = torch_traffich.numpy()
    return trafficl, traffich

def stl_decomposition(series):
    """
    对ndarray的每一列进行STL分解，返回原始序列、趋势项和季节项
    
    Args:
    series: ndarray, shape (n_samples, n_features)
    
    Returns:
    original_series: ndarray, shape (n_samples, n_features)
    trend_series: ndarray, shape (n_samples, n_features)
    seasonal_series: ndarray, shape (n_samples, n_features)
    """
    trend_series = np.zeros(series.shape)
    seasonal_series = np.zeros(series.shape)

    # 只需直接返回原始序列
    original_series = series

    for i in range(series.shape[1]):
        # 对每一列进行STL分解
        stl = STL(series[:, i],period=10,robust=False)
        res = stl.fit()

        # 获取趋势项和季节项
        trend_series[:, i] = res.trend
        seasonal_series[:, i] = res.seasonal

    return original_series, trend_series, seasonal_series


# def vmd_decomposition(series, K=3, alpha=200, tau=0.1, tol=1e-6):
#     """
#     对ndarray的每一列进行VMD分解，返回原始序列、趋势项和季节项
    
#     Args:
#     series: ndarray, shape (n_samples, n_features)
#     K: int, 分解的模态数量，默认为2
#     alpha: float, 带宽约束参数
#     tau: float, 容忍噪声参数
#     tol: float, 收敛容忍度
    
#     Returns:
#     original_series: ndarray, shape (n_samples, n_features)
#     trend_series: ndarray, shape (n_samples, n_features)
#     seasonal_series: ndarray, shape (n_samples, n_features)
#     """
#     trend_series = np.zeros(series.shape)
#     seasonal_series = np.zeros(series.shape)

#     # 只需直接返回原始序列
#     original_series = series

#     for i in range(series.shape[1]):
#         # 对每一列进行VMD分解
#         u, _, _ = VMD(series[:, i], alpha, tau, K, 0, 1, tol)

#         # 确保模态数组与原始序列长度一致
#         len_diff = series.shape[0] - u.shape[1]
#         if len_diff > 0:
#             # 如果原始序列更长，填充模态数组
#             u = np.pad(u, ((0, 0), (0, len_diff)), mode='constant', constant_values=0)
#         elif len_diff < 0:
#             # 如果模态数组更长，截断模态数组
#             u = u[:, :series.shape[0]]

#         # 获取趋势项和季节项
#         trend_series[:, i] = u[0] if K > 0 else np.zeros_like(series[:, i])
#         seasonal_series[:, i] = u[1] if K > 1 else np.zeros_like(series[:, i])

#     return original_series, trend_series, seasonal_series




def loadData(args):
    # Traffic
    # Traffic = np.squeeze(np.load(args.traffic_file)['result'], -1)
    Traffic = np.load(args.traffic_file)['result']
    print(Traffic.shape)
    # train/val/test 
    num_step = Traffic.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]

    # 趋势项和季节项提取
    ## train
    original_train, trend_train, seasonal_train = stl_decomposition(train)
    ## val
    original_train, trend_val, seasonal_val = stl_decomposition(val)
    ## test
    original_train, trend_test, seasonal_test = stl_decomposition(test)

    # X, Y
    trainX, trainY = seq2instance(train, args.T1, args.T2)
    valX, valY = seq2instance(val, args.T1, args.T2)
    testX, testY = seq2instance(test, args.T1, args.T2)
    ## train
    trainXL, trainYL = seq2instance(trend_train, args.T1, args.T2)
    trainXH, trainYH = seq2instance(seasonal_train, args.T1, args.T2)
    ## val
    valXL, valYL = seq2instance(trend_val, args.T1, args.T2)
    valXH, valYH = seq2instance(seasonal_val, args.T1, args.T2)    
    ## test
    testXL, testYL = seq2instance(trend_test, args.T1, args.T2)
    testXH, testYH = seq2instance(seasonal_test, args.T1, args.T2) 

    # X, Y
    # trainX, trainY = seq2instance(train, args.T1, args.T2)
    # valX, valY = seq2instance(val, args.T1, args.T2)
    # testX, testY = seq2instance(test, args.T1, args.T2)
    mean, std = np.mean(trainX), np.std(trainX)
    trainXL, trainXH = (trainXL - mean) / std, (trainXH - mean) / std
    valXL, valXH = (valXL - mean) / std, (valXH - mean) / std
    testXL, testXH = (testXL - mean) / std, (testXH - mean) / std
    trainX, valX, testX = (trainX - mean) / std, (valX - mean) / std, (testX - mean) / std
    # temporal embedding
    tmp = {'PeMSD3':6,'PeMSD4':1,'PeMSD7':1,'PeMSD8':5, 'PeMSD7L':2, 'PeMSD7M':2, 'MYDATA':1}
    days = {'PeMSD3':7,'PeMSD4':7,'PeMSD7':7,'PeMSD8':7, 'PeMSD7L':5, 'PeMSD7M':5, 'MYDATA':4}
    TE = np.zeros([num_step, 2])


    ######### My data##############
    startd = (tmp[args.Dataset] - 1) * 3
    df = days[args.Dataset]
    startt = 0
    for i in range(num_step):
        TE[i,0] = startd //  3
        startd = (startd + 1) % (df * 3)
        TE[i,1] = startt
        startt = (startt + 1) % 3
    ##############


    # train/val/test
    train = TE[: train_steps]
    val = TE[train_steps : train_steps + val_steps]
    test = TE[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.T1, args.T2)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.T1, args.T2)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.T1, args.T2)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)
    
    return trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, testXL, testXH, testTE, testY, mean, std


import torch
from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    def __init__(self, args, mode, mean=None, std=None):
        self.args = args
        self.mode = mode
        self.Traffic = np.load(args.traffic_file)['result']
        self.num_step = self.Traffic.shape[0]
        self.train_steps = round(args.train_ratio * self.num_step)
        self.test_steps = round(args.test_ratio * self.num_step)
        self.val_steps = self.num_step - self.train_steps - self.test_steps
        self.mean, self.std = None, None

        if mode == 'train':
            self.data = self.Traffic[: self.train_steps]
        elif mode == 'val':
            self.data = self.Traffic[self.train_steps : self.train_steps + self.val_steps]
        elif mode == 'test':
            self.data = self.Traffic[-self.test_steps :]
        
        self.original, self.trend, self.seasonal = stl_decomposition(self.data)
        self.X, self.Y = seq2instance(self.data, args.T1, args.T2)
        self.XL, self.YL = seq2instance(self.trend, args.T1, args.T2)
        self.XH, self.YH = seq2instance(self.seasonal, args.T1, args.T2)

        if mode == 'train':
            self.mean, self.std = np.mean(self.X), np.std(self.X)
        else:
            self.mean, self.std = mean, std

        self.XL, self.XH = (self.XL - self.mean) / self.std, (self.XH - self.mean) / self.std
        self.X = (self.X - self.mean) / self.std

        # Compute temporal embedding
        self.TE = self.compute_te(args, self.num_step)

        if mode == 'train':
            self.te_data = self.TE[: self.train_steps]
        elif mode == 'val':
            self.te_data = self.TE[self.train_steps : self.train_steps + self.val_steps]
        elif mode == 'test':
            self.te_data = self.TE[-self.test_steps :]

        self.TE = seq2instance(self.te_data, args.T1, args.T2)
        self.TE = np.concatenate(self.TE, axis = 1).astype(np.int32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # x = {
        #     'XL': torch.tensor(self.XL[index]),
        #     'XH': torch.tensor(self.XH[index]),
        #     'TE': torch.tensor(self.TE[index]),
        # }
        x = (torch.tensor(self.X[index]), torch.tensor(self.XL[index]), torch.tensor(self.XH[index]), torch.tensor(self.TE[index]))
        y = (torch.tensor(self.Y[index]), torch.tensor(self.YL[index]))
        # y = {
        # 'Y': torch.tensor(self.Y[index]),
        # 'YL': torch.tensor(self.YL[index]),
        # }
        return x, y

    # def __getitem__(self, index):
    #     return (torch.tensor(self.XL[index]), torch.tensor(self.XH[index]), torch.tensor(self.TE[index]), 
    #             torch.tensor(self.Y[index]), torch.tensor(self.YL[index]))


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
    
# 模型中，你可以从标签中提取出所需的信息。
# for inputs, labels in dataloader:
#     XL = inputs['XL']
#     XH = inputs['XH']
#     TE = inputs['TE']
#     Y, YL = labels
#     # 在这里，你可以使用XL, XH, TE, Y, YL来训练你的模型


class MetaDataset(Dataset):
    """

    **Description**

    Wraps a classification dataset to enable fast indexing of samples within classes.

    This class exposes two attributes specific to the wrapped dataset:

    * `labels_to_indices`: maps a class label to a list of sample indices with that label.
    * `indices_to_labels`: maps a sample index to its corresponding class label.

    Those dictionary attributes are often used to quickly create few-shot classification tasks.
    They can be passed as arguments upon instantiation, or automatically built on-the-fly.
    If the wrapped dataset has an attribute `_bookkeeping_path`, then the built attributes will be cached on disk and reloaded upon the next instantiation.
    This caching strategy is useful for large datasets (e.g. ImageNet-1k) where the first instantiation can take several hours.

    Note that if only one of `labels_to_indices` or `indices_to_labels` is provided, this class builds the other one from it.

    **Arguments**

    * **dataset** (Dataset) -  A torch Dataset.
    * **labels_to_indices** (dict, **optional**, default=None) -  A dictionary mapping labels to the indices of their samples.
    * **indices_to_labels** (dict, **optional**, default=None) -  A dictionary mapping sample indices to their corresponding label.

    **Example**
    ~~~python
    mnist = torchvision.datasets.MNIST(root="/tmp/mnist", train=True)
    mnist = l2l.data.MetaDataset(mnist)
    ~~~
    """

    def __init__(self, dataset, labels_to_indices=None, indices_to_labels=None):

        if not isinstance(dataset, Dataset):
            raise TypeError(
                "MetaDataset only accepts a torch dataset as input")

        self.dataset = dataset

        if hasattr(dataset, '_bookkeeping_path'):
            self.load_bookkeeping(dataset._bookkeeping_path)
        else:
            self.create_bookkeeping(
                labels_to_indices=labels_to_indices,
                indices_to_labels=indices_to_labels,
            )

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def create_bookkeeping(self, labels_to_indices=None, indices_to_labels=None):
        """
        Iterates over the entire dataset and creates a map of target to indices.

        Returns: A dict with key as the label and value as list of indices.
        """

        assert hasattr(self.dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

        # Bootstrap from arguments
        if labels_to_indices is not None:
            indices_to_labels = {
                idx: label
                for label, indices in labels_to_indices.items()
                for idx in indices
            }
        elif indices_to_labels is not None:
            labels_to_indices = defaultdict(list)
            for idx, label in indices_to_labels.items():
                labels_to_indices[label].append(idx)
        else:  # Create from scratch
            labels_to_indices = defaultdict(list)
            indices_to_labels = defaultdict(int)
            for i in range(len(self.dataset)):
                try:
                    label = self.dataset[i][1]
                    # if label is a Tensor, then take get the scalar value
                    if hasattr(label, 'item'):
                        label = self.dataset[i][1].item()
                except ValueError as e:
                    raise ValueError(
                        'Requires scalar labels. \n' + str(e))

                labels_to_indices[label].append(i)
                indices_to_labels[i] = label

        self.labels_to_indices = labels_to_indices
        self.indices_to_labels = indices_to_labels
        self.labels = list(self.labels_to_indices.keys())

        self._bookkeeping = {
            'labels_to_indices': self.labels_to_indices,
            'indices_to_labels': self.indices_to_labels,
            'labels': self.labels
        }

    def load_bookkeeping(self, path):
        if not os.path.exists(path):
            self.create_bookkeeping()
            self.serialize_bookkeeping(path)
        else:
            with open(path, 'rb') as f:
                self._bookkeeping = pickle.load(f)
            self.labels_to_indices = self._bookkeeping['labels_to_indices']
            self.indices_to_labels = self._bookkeeping['indices_to_labels']
            self.labels = self._bookkeeping['labels']

    def serialize_bookkeeping(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._bookkeeping, f, protocol=-1)

class MetaTrafficDataset(TrafficDataset, MetaDataset):
    def __init__(self, *args, **kwargs):
        TrafficDataset.__init__(self, *args, **kwargs)
        MetaDataset.__init__(self, self)

    def __getitem__(self, index):
        x, y = TrafficDataset.__getitem__(self, index)
        return x, y

    def __len__(self):
        return TrafficDataset.__len__(self)


def extract_fields(dataset):
    XL, XH, TE, Y, YL = [], [], [], [], []
    for inputs, labels in dataset:
        XL.append(inputs[0].numpy())
        XH.append(inputs[1].numpy())
        TE.append(inputs[2].numpy())
        Y.append(labels[0].numpy())
        YL.append(labels[1].numpy())
    return np.stack(XL), np.stack(XH), np.stack(TE), np.stack(Y), np.stack(YL)

from collections import defaultdict
from learn2learn.data.transforms import TaskTransform
from learn2learn.data.task_dataset import DataDescription
import random

# class MyFusedNWaysKShots(TaskTransform):

#     def __init__(self, dataset, meta_bsz):
#         super(MyFusedNWaysKShots, self).__init__(dataset)
#         self.meta_bsz = meta_bsz

#     def new_task(self):
#         task_size = len(self.dataset) // self.meta_bsz
#         task_description = []
#         for i in range(0, len(self.dataset), task_size):
#             if i + task_size > len(self.dataset):  # If the remaining tasks are less than task_size, break the loop
#                 break
#             for j in range(i, i + task_size):
#                 dd = DataDescription(j)
#                 dd.transforms.append(lambda x: self.dataset[x])  # Add a transform to load the data
#                 task_description.append(dd)
#         return task_description 
    
#     def __call__(self, task_description):
#         if task_description is None:
#             task_description = self.new_task()
#         task_description = [DataDescription(dd.index) for dd in task_description]
#         return task_description


class MyFusedNWaysKShots(TaskTransform):

    def __init__(self, dataset, meta_bsz):
        super(MyFusedNWaysKShots, self).__init__(dataset)
        self.meta_bsz = meta_bsz

    def new_task(self):
        task_size = len(self.dataset) // self.meta_bsz
        task_description = []
        for i in range(0, len(self.dataset), task_size):
            if i + task_size > len(self.dataset):  # If the remaining tasks are less than task_size
                remaining = len(self.dataset) - i
                for j in range(i, i + remaining):
                    dd = DataDescription(j)
                    dd.transforms.append(lambda x: self.dataset[x])  # Add a transform to load the data
                    task_description.append(dd)
                # Randomly sample tasks to fill the last batch
                for j in random.sample(range(len(self.dataset)), task_size - remaining):
                    dd = DataDescription(j)
                    dd.transforms.append(lambda x: self.dataset[x])  # Add a transform to load the data
                    task_description.append(dd)
                break
            for j in range(i, i + task_size):
                dd = DataDescription(j)
                dd.transforms.append(lambda x: self.dataset[x])  # Add a transform to load the data
                task_description.append(dd)
        return task_description

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        task_description = [DataDescription(dd.index) for dd in task_description]
        return task_description

# def plot_metrics(maes, rmses, mapes, epochs, picture_dir):
#     fig, axs = plt.subplots(3, figsize=(12, 10))  # 调整 figsize 参数来改变图窗的尺寸
#     fig.suptitle('Average Metrics over Iterations')
#     epochs = list(range(1, len(maes) + 1))
#     axs[0].plot(epochs, maes, label='MAE')
#     axs[0].set(xlabel='Iteration', ylabel='MAE')
#     axs[0].legend()

#     axs[1].plot(epochs, rmses, label='RMSE')
#     axs[1].set(xlabel='Iteration', ylabel='RMSE')
#     axs[1].legend()

#     axs[2].plot(epochs, mapes, label='MAPE')
#     axs[2].set(xlabel='Iteration', ylabel='MAPE')
#     axs[2].legend()

#     plt.savefig(os.path.join(picture_dir, 'metrics.png'))
#     # plt.show()

def plot_metrics(valid_maes, valid_rmses, valid_mapes, test_maes, test_rmses, test_mapes, iterations, picture_dir):
    fig, axs = plt.subplots(3, figsize=(12, 10))  # adjust figsize to change the size of the figure
    fig.suptitle('Average Metrics over Iterations')
    iterations = list(range(1, len(valid_maes) + 1))
    
    axs[0].plot(iterations, valid_maes, label='Valid MAE')
    axs[0].plot(iterations, test_maes, label='Test MAE')
    axs[0].set(xlabel='Iteration', ylabel='MAE')
    axs[0].legend()

    axs[1].plot(iterations, valid_rmses, label='Valid RMSE')
    axs[1].plot(iterations, test_rmses, label='Test RMSE')
    axs[1].set(xlabel='Iteration', ylabel='RMSE')
    axs[1].legend()

    axs[2].plot(iterations, valid_mapes, label='Valid MAPE')
    axs[2].plot(iterations, test_mapes, label='Test MAPE')
    axs[2].set(xlabel='Iteration', ylabel='MAPE')
    axs[2].legend()

    plt.savefig(os.path.join(picture_dir, 'metrics.png'))
    # plt.show()


def save_to_csv(data, file_path):
    # 写入 CSV 文件
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    print(f"已保存数据到 {file_path}")
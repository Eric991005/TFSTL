from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
import time
import argparse
import configparser
import csv
import math
import random
# from pytorch_wavelets import DWT1DForward, DWT1DInverse

from lib import utils_meta
from lib.utils_meta import log_string, loadData, _compute_loss, metric, TrafficDataset
from lib.graph_utils import loadGraph
from mymodel.models_meta import TFSTL
import os
from torch.utils.tensorboard import SummaryWriter

tensorboard_folder = '/root/autodl-tmp/MYSTWave/runs/STL_export'

# 检查并创建TensorBoard文件夹
if not os.path.exists(tensorboard_folder):
    os.makedirs(tensorboard_folder)
    print(f"创建了文件夹：{tensorboard_folder}")
else:
    print(f"文件夹已存在：{tensorboard_folder}")

tensor_writer = SummaryWriter(tensorboard_folder)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)

parser.add_argument('--cuda', type=str, 
            default=config['train']['cuda'])
parser.add_argument('--seed', type = int, 
            default = config['train']['seed'])
parser.add_argument('--batch_size', type = int, 
            default = config['train']['batch_size'])
parser.add_argument('--max_epoch', type = int, 
            default = config['train']['max_epoch'])
parser.add_argument('--learning_rate', type=float, 
            default = config['train']['learning_rate'])

parser.add_argument('--Dataset', default = config['data']['dataset'])
parser.add_argument('--T1', type = int, 
            default = config['data']['T1'])
parser.add_argument('--T2', type = int, 
            default = config['data']['T2'])
parser.add_argument('--train_ratio', type = float, 
            default = config['data']['train_ratio'])
parser.add_argument('--val_ratio', type = float, 
            default = config['data']['val_ratio'])
parser.add_argument('--test_ratio', type = float, 
            default = config['data']['test_ratio'])

parser.add_argument('--L', type = int,
            default = config['param']['layers'])
parser.add_argument('--h', type = int,
            default = config['param']['heads'])
parser.add_argument('--d', type = int, 
            default = config['param']['dims'])
parser.add_argument('--j', type = int, 
            default = config['param']['level'])
parser.add_argument('--s', type = float,
            default = config['param']['samples'])
parser.add_argument('--w',
            default = config['param']['wave'])

parser.add_argument('--traffic_file', default = config['file']['traffic'])
parser.add_argument('--adj_file', default = config['file']['adj'])
parser.add_argument('--adjgat_file', default = config['file']['adjgat'])
parser.add_argument('--model_file', default = config['file']['model'])
parser.add_argument('--log_file', default = config['file']['log'])

args = parser.parse_args()

log = open(args.log_file, 'w')

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

def res(model, valXL, valXH, valTE, valY, mean, std, adjgat):
    model.eval()
    num_val = valXL.shape[0]
    num_batch = math.ceil(num_val / args.batch_size)

    pred = []
    label = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(valXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(valXH[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)

                y_hat, y_hat_l = model(xl, xh, te, adjgat)

                pred.append(y_hat.cpu().numpy()*std+mean)
                label.append(y)
    
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)

    maes = []
    rmses = []
    mapes = []

    for i in range(pred.shape[1]):
        mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
    
    mae, rmse, mape = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape, ))
    
    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

def test_res(model, valXL, valXH, valTE, valY, mean, std, adjgat):
    model.eval()
    num_val = valXL.shape[0]
    num_batch = math.ceil(num_val / args.batch_size)

    pred = []
    label = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(valXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(valXH[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)

                y_hat, y_hat_l = model(xl, xh, te, adjgat)

                pred.append(y_hat.cpu().numpy()*std+mean)
                label.append(y)
    
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)

    maes = []
    rmses = []
    mapes = []

    for i in range(pred.shape[1]):
        mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
    
    mae, rmse, mape = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))

    # 从 NumPy 数组中提取数据
    data = pred[:, 0, :]

    # 写入 CSV 文件
    file_path = '/root/autodl-tmp/MYSTWave/output/export_pred_out/first_pred.csv'
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    print(f"已保存数据到 {file_path}")

    # 从 NumPy 数组中提取数据
    data = pred[:, 1, :]

    # 写入 CSV 文件
    file_path = '/root/autodl-tmp/MYSTWave/output/export_pred_out/second_pred.csv'
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    print(f"已保存数据到 {file_path}")

    # 从 NumPy 数组中提取数据
    data = label[:, 0, :]

    # 写入 CSV 文件
    file_path = '/root/autodl-tmp/MYSTWave/output/export_pred_out/first_label.csv'
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    print(f"已保存数据到 {file_path}")

    # 从 NumPy 数组中提取数据
    data = label[:, 1, :]

    # 写入 CSV 文件
    file_path = '/root/autodl-tmp/MYSTWave/output/export_pred_out/second_label.csv'
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    print(f"已保存数据到 {file_path}")

    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

def train(model, trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, mean, std, adjgat):
    num_train = trainXL.shape[0]
    # min_loss = 2000000000.0
    min_loss = 0.5
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200,    
                                    verbose=False, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=2e-6, eps=1e-08)
    
    for epoch in tqdm(range(1,args.max_epoch+1)):
        model.train()
        train_l_sum, batch_count, start = 0.0, 0, time.time()
        permutation = np.random.permutation(num_train)
        trainXL = trainXL[permutation]
        trainXH = trainXH[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        trainYL = trainYL[permutation]
        num_batch = math.ceil(num_train / args.batch_size)

        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(trainXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(trainXH[start_idx : end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx : end_idx]).float().to(device)
                yl = torch.from_numpy(trainYL[start_idx : end_idx]).float().to(device)
                te = torch.from_numpy(trainTE[start_idx : end_idx]).to(device)
                
                
                optimizer.zero_grad()

                y_hat, y_hat_l = model(xl, xh, te, adjgat)

                loss = _compute_loss(y, y_hat*std+mean) + _compute_loss(yl, y_hat_l*std+mean)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                train_l_sum += loss.cpu().item()
                batch_count += 1
                pbar.update(1)

        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        tensor_writer.add_scalar('training loss', train_l_sum / batch_count, epoch)

        mae, rmse, mape = res(model, valXL, valXH, valTE, valY, mean, std, adjgat)
        lr_scheduler.step(mape[-1])
        if mape[-1] < min_loss:
            min_loss = mape[-1]
            torch.save(model.state_dict(), args.model_file)

def test(model, valXL, valXH, valTE, valY, mean, std, adjgat):
    try:
        model.load_state_dict(torch.load(args.model_file))
    except EOFError:
        print(f"Error: Unable to load model state dictionary from file {args.model_file}. File may be empty or corrupted.")
        return

    mae, rmse, mape = test_res(model, valXL, valXH, valTE, valY, mean, std, adjgat)
    return mae, rmse, mape

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)




if __name__ == '__main__':
    log_string(log, "loading data....")
    train_dataset = TrafficDataset(args, 'train')
    val_dataset = TrafficDataset(args, 'val', train_dataset.mean, train_dataset.std)
    test_dataset = TrafficDataset(args, 'test', train_dataset.mean, train_dataset.std)
    adjgat = loadGraph(args)
    adjgat = torch.from_numpy(adjgat).float().to(device)
    log_string(log, "loading end....")

    log_string(log, "constructing model begin....")
    # Create model
    model = TFSTL(1, args.h*args.d, args.L, args.h, args.d, args.s, args.T1, args.T2, device).to(device)
    log_string(log, "constructing model end....")

    log_string(log, "training begin....")
    train(model, train_dataset.XL, train_dataset.XH, train_dataset.TE, train_dataset.Y, train_dataset.YL, val_dataset.XL, val_dataset.XH, val_dataset.TE, val_dataset.Y, train_dataset.mean, train_dataset.std, adjgat)
    log_string(log, "training end....")

    log_string(log, "testing begin....")
    test(model, test_dataset.XL, test_dataset.XH, test_dataset.TE, test_dataset.Y, train_dataset.mean, train_dataset.std, adjgat)
    log_string(log, "testing end....")

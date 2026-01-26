# train.py
import torch
import os
import argparse
import time
import json
import torch.nn as nn
from FishTrackDataset import get_dataset
from model import LSTMModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import csv
import joblib

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    experiment_dir = f"./exp/exp_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)

    with open(experiment_dir+'/Config.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # 创建错误记录文件
    error_csv = os.path.join(experiment_dir, 'error_samples.csv')
    with open(error_csv, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Epoch', 'Phase', 'Filename', 'TrackID', 'Predicted', 'Actual'])

    # 获取数据集，假设get_dataset返回2个值
    train_loader, val_loader = get_dataset(args.data_path, fps=args.fps, batch_size=args.batch_size)
    
    # 获取数据集对象以访问文件名和ID信息
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = LSTMModel(args.input_size, args.hidden_size, args.num_layers, args.classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999])
    elif args.opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    tb_writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tensorboard'))  # 重命名TensorBoard writer
    best_accuracy = 0.0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_loss, train_acc, train_errors = train(
            train_loader, model, criterion, optimizer, device, train_dataset, epoch
        )
        
        # 记录训练错误样本
        with open(error_csv, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            for filename, track_id, pred, actual in train_errors:
                csv_writer.writerow([epoch, 'train', filename, track_id, pred, actual])

        scheduler.step()
        val_loss, val_acc, val_errors = evaluate(
            val_loader, model, criterion, device, val_dataset, epoch
        )
        
        # 记录验证错误样本
        with open(error_csv, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            for filename, track_id, pred, actual in val_errors:
                csv_writer.writerow([epoch, 'val', filename, track_id, pred, actual])
        
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch:{epoch + 1}, time:{epoch_duration:.2f}s")
        tb_writer.add_scalar('training loss', train_loss, epoch)  # 使用重命名后的TensorBoard writer
        tb_writer.add_scalar('training acc', train_acc, epoch)
        tb_writer.add_scalar('val loss', val_loss, epoch)
        tb_writer.add_scalar('val acc', val_acc, epoch)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            print(f"best_accuracy epoch{epoch + 1}, save model")
            torch.save(model.state_dict(), os.path.join(experiment_dir, f'model_epoch_{epoch+1}.pth'))
            
    scaler_path = os.path.join(experiment_dir, 'scaler.pkl')
    joblib.dump(train_loader.dataset.get_scaler(), scaler_path)

    print(f"标准化器已保存至 {scaler_path}")

    torch.save(model.state_dict(), os.path.join(experiment_dir, 'model_last.pth'))
    print("最终模型已保存为 'model_last.pth'")
    tb_writer.close()  # 使用重命名后的TensorBoard writer

def train(data_loader, model, criterion, optimizer, device, dataset, epoch):
    model.train()
    training_loss = 0.0
    num_correct = 0
    num_total = 0
    error_samples = []  # 存储错误样本信息

    for batch_idx, (tracks, lengths, labels) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1} Training")):
        tracks, lengths, labels = tracks.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(tracks, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        predicts = outputs.max(dim=1)[1]
        
        # 记录错误样本
        for i in range(len(labels)):
            if predicts[i] != labels[i]:
                # 计算原始数据集中样本的索引
                sample_idx = batch_idx * data_loader.batch_size + i
                if sample_idx >= len(dataset):
                    continue
                
                # 假设数据集有获取文件名和ID的方法
                # 如果数据集没有这些方法，需要根据实际情况修改
                filename = getattr(dataset, 'get_filename', lambda x: f"file_{x}")(sample_idx)
                track_id = getattr(dataset, 'get_track_id', lambda x: x)(sample_idx)
                
                error_samples.append((filename, track_id, predicts[i].item(), labels[i].item()))
        
        num_correct += (predicts == labels).sum().item()
        num_total += len(labels)

    training_loss /= len(data_loader)
    training_acc = 100.0 * num_correct / num_total
    print(f'Training set: Average loss: {training_loss:.4f}, Accuracy: {num_correct}/{num_total} ({training_acc:.2f}%)')
    return training_loss, training_acc, error_samples

def evaluate(data_loader, model, criterion, device, dataset, epoch):
    model.eval()
    val_loss = 0.0
    num_correct = 0
    num_total = 0
    error_samples = []  # 存储错误样本信息

    with torch.no_grad():
        for batch_idx, (tracks, lengths, labels) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1} Validation")):
            tracks, lengths, labels = tracks.to(device), lengths.to(device), labels.to(device)
            outputs = model(tracks, lengths)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicts = outputs.max(dim=1)[1]
            
            # 记录错误样本
            for i in range(len(labels)):
                if predicts[i] != labels[i]:
                    # 计算原始数据集中样本的索引
                    sample_idx = batch_idx * data_loader.batch_size + i
                    if sample_idx >= len(dataset):
                        continue
                    
                    # 假设数据集有获取文件名和ID的方法
                    # 如果数据集没有这些方法，需要根据实际情况修改
                    filename = getattr(dataset, 'get_filename', lambda x: f"file_{x}")(sample_idx)
                    track_id = getattr(dataset, 'get_track_id', lambda x: x)(sample_idx)
                    
                    error_samples.append((filename, track_id, predicts[i].item(), labels[i].item()))
            
            num_correct += (predicts == labels).sum().item()
            num_total += len(labels)

    val_loss /= len(data_loader)
    val_acc = 100.0 * num_correct / num_total
    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {num_correct}/{num_total} ({val_acc:.2f}%)')
    return val_loss, val_acc, error_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for track learning")
    parser.add_argument('--data_path', type=str, default='/home/bwang/FishTrack_5/dataset_z', help="Data path")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, help="Number of samples per batch")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--opt', type=str, default='Adam', help="Optimizer, SGD or Adam")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--fps', type=float, default=25.0, help="Frames per second")
    parser.add_argument('--input_size', type=int, default=5, help="Features of each input (x,y,v,cos,sin)")
    parser.add_argument('--hidden_size', type=int, default=128, help="Hidden numbers of LSTM")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of stacked LSTMs")
    parser.add_argument('--classes', type=int, default=2, help="Classes of fish")
    args = parser.parse_args()
    main(args)
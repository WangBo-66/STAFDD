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

    train_loader, val_loader = get_dataset(args.data_path, fps=args.fps, batch_size=args.batch_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = LSTMModel(args.input_size, args.hidden_size, args.num_layers, args.classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999])
    elif args.opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tensorboard'))
    best_accuracy = 0.0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_acc = evaluate(val_loader, model, criterion, device)
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch:{epoch + 1}, time:{epoch_duration:.2f}s")
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training acc', train_acc, epoch)
        writer.add_scalar('val loss', val_loss, epoch)
        writer.add_scalar('val acc', val_acc, epoch)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            print(f"best_accuracy epoch{epoch + 1}, save model")
            torch.save(model.state_dict(), os.path.join(experiment_dir, f'model_epoch_{epoch+1}.pth'))
            
    import joblib
    scaler_path = os.path.join(experiment_dir, 'scaler.pkl')
    joblib.dump(train_loader.dataset.get_scaler(), scaler_path)

    print(f"标准化器已保存至 {scaler_path}")

    torch.save(model.state_dict(), os.path.join(experiment_dir, 'model_last.pth'))
    print("最终模型已保存为 'model_last.pth'")
    writer.close()

def train(data_loader, model, criterion, optimizer, device):
    model.train()
    training_loss = 0.0
    num_correct = 0
    num_total = 0

    for tracks, lengths, labels in tqdm(data_loader):
        tracks, lengths, labels = tracks.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(tracks, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        predicts = outputs.max(dim=1)[1]
        num_correct += (predicts == labels).sum().item()
        num_total += len(labels)

    training_loss /= len(data_loader)
    training_acc = 100.0 * num_correct / num_total
    print(f'Training set: Average loss: {training_loss:.4f}, Accuracy: {num_correct}/{num_total} ({training_acc:.2f}%)')
    return training_loss, training_acc

def evaluate(data_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for tracks, lengths, labels in tqdm(data_loader):
            tracks, lengths, labels = tracks.to(device), lengths.to(device), labels.to(device)
            outputs = model(tracks, lengths)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicts = outputs.max(dim=1)[1]
            num_correct += (predicts == labels).sum().item()
            num_total += len(labels)

    val_loss /= len(data_loader)
    val_acc = 100.0 * num_correct / num_total
    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {num_correct}/{num_total} ({val_acc:.2f}%)')
    return val_loss, val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for track learning")
    parser.add_argument('--data_path', type=str, default='/home/bwang/FishTrack_5/dataset', help="Data path")
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

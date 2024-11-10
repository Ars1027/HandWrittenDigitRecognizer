import torch
import numpy as np
import argparse
from tqdm import tqdm

import torch.utils # 命令行处理参数
import torch.nn
from dataset import MNIST_dataset
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from classifier import SimpleCNN
# from classifier import MLP
import utils

def get_args_parser():
    # python train.py -train_data_path /.. /.. / -lr 1e-6 -bs 16 -e 8 训练输入的参数
    parser = argparse.ArgumentParser()
    # 路径
    parser.add_argument(
        "-train_data_path", 
        dest = "train_data_path",
        default = "C:\\Users\\eason\\Desktop\\Learning Materials\\Third year\\MyPthonCourse\\myItem\\MNIST_Parsed\\MNIST_Parsed\\train",
        type = str,
        help = "Path to training data."
    )
    # 学习率
    parser.add_argument(
        "-lr", 
        dest = "learning_rate",
        default = 1e-5,
        type = float,
        help = "Learing rate."
    )
    # 样本数量
    parser.add_argument(
        "-bs", 
        dest = "batch_size",
        default = 16,
        type = int,
        help = "The batch size."
    )
    # 迭代次数
    parser.add_argument(
        "-e", 
        dest = "epoch",
        default = 10,
        type = int,
        help = "The traing epoch."
    )
    
    return parser.parse_args()

def train(train_dataloader, val_dataloader, args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(nb_class=10).to(device) # 模型选择
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5) # 优化参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss() # 损失函数
    
    for epoch in range(args.epoch):
        train_loss_sum = 0
        # 每一次训练一个batch
        for batch_idx, (X, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # forward 神经网络前传（训练）
            X = X.to(device)
            y = y.to(device)
            pred = model(X).to(device)
            loss = loss_fn(pred, y)
            
            # backward 向后传播(计算并优化损失函数的误差)
            loss.backward()
            optimizer.step() # 计算步长(（i梯度）
            optimizer.zero_grad() # 清空梯度
            
            train_loss_sum += loss.item() / len(train_dataloader) # 累加每一个batch的loss并计算总的loss
            
        print("Epoch ()/{}".format(epoch + 1, args.epoch))
        print("Loss: {}".format(train_loss_sum))
        
        print("Accuracy on training set:")
        utils.check_accuracy(train_dataloader, model, device)
        print("Accuracy on validation set:")
        utils.check_accuracy(val_dataloader, model, device)
            
    
    return model

# # 验证集测试
# def check_accuracy(data_loader, model, device):
#     accuracy = 0
#     num_correct = 0
#     num_sample = 0
#     model.eval()
    
#     with torch.no_grad():
#         for batch_idx, (X, y) in enumerate(data_loader):
#             # forward
#             X = X.to(device)
#             y = y.to(device)
            
#             pred = model(X) # pred是一个向量[]，代表所有数字的预测概率
#             y_hat = torch.argmax(pred, dim=1)
#             num_correct += (y_hat == y).sum() # 判断预测值正确的数量
#             num_sample += pred.shape[0] # 样本数量
        
#     accuracy = float(num_correct)/float(num_sample)
#     print("{}/{}, Accuracy: {}".format(num_correct, num_sample, accuracy))
    
#     return accuracy
    

if __name__ == "__main__":
    args = get_args_parser()
    # print(args.epoch)
    
    # train_dataset = MNIST_dataset("C:\\Users\\eason\\Desktop\\Learning Materials\\Third year\\MyPthonCourse\\myItem\\train\\MNIST_Parsed\\MNIST_Parsed\\train_01")
    train_dataset = MNIST_dataset(args.train_data_path)
    
    # 训练集和验证集的长度
    train_len = int(len(train_dataset) * 0.8)
    val_len = len(train_dataset) - train_len
    
    generator = torch.Generator().manual_seed(43)
    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len], generator) # 分割训练数据：8训练集:2验证集
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        # batch_size=16,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader =  DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        # batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    
    model = train(train_dataloader, val_dataloader, args)
    
    # torch.save(model.state_dict(), "model.weigth.pth") # 只保存模型参数
    torch.save(model, "model.pth") # 保存模型（结构+参数）
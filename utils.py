import torch

# 验证集测试
def check_accuracy(data_loader, model, device):
    accuracy = 0
    num_correct = 0
    num_sample = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            # forward
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X) # pred是一个向量[]，代表所有数字的预测概率
            y_hat = torch.argmax(pred, dim=1)
            num_correct += (y_hat == y).sum() # 判断预测值正确的数量
            num_sample += pred.shape[0] # 样本数量
            print(y, y_hat)
        
    accuracy = float(num_correct)/float(num_sample)
    print("{}/{}, Accuracy: {}".format(num_correct, num_sample, accuracy))
    
    return accuracy
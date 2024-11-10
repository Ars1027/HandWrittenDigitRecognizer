import argparse
import torch

from dataset import MNIST_dataset
from torch.utils.data import DataLoader
import utils

def get_args_parser():
    
    parser = argparse.ArgumentParser()
    
    # 测试集路径
    parser.add_argument(
        "-test_data_path", 
        dest = "test_data_path",
        default = "C:\\Users\\eason\\Desktop\\Learning Materials\\Third year\\MyPthonCourse\\myItem\\MNIST_Parsed\\MNIST_Parsed\\test",
        type = str,
        help = "Path to test data."
    )
    
    # 模型路径
    parser.add_argument(
        "-model_path", 
        dest = "model_path",
        default = "C:\\Users\\eason\\Desktop\\Learning Materials\\Third year\\MyPthonCourse\\myItem\\model.pth",
        type = str,
        help = "Path to the model."
    )
    
    # 样本数量
    parser.add_argument(
        "-bs", 
        dest = "batch_size",
        default = 16,
        type = int,
        help = "The batch size."
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args_parser()
    
    test_dataset = MNIST_dataset(args.test_data_path)
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path).to(device) # 读模型
    
    accuracy = utils.check_accuracy(
        data_loader=test_dataloader,
        model=model,
        device=device  
    )
    
    
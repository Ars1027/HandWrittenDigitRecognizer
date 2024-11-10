import numpy as np
import glob
import os
import random
import cv2
from torch.utils.data import Dataset


class MNIST_dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_dirs = [] # 所有图片的路径
        self.all_lables = [] #所有图片的标签
        
        self._loadDataDir()
        
    def _loadDataDir(self):
        self.all_dirs = glob.glob(
            # "C:\\Users\\eason\\Desktop\\学习\\大三\\python开发与实训\\myItem\\torch\\MNIST_data\\MNIST_data\\train\\**\\*.png"
            os.path.join(
                os.path.join(self.path, "**"), "*.png"
            )
        )
        self.all_lables = [path.split(os.path.sep)[-2] for path in self.all_dirs]
        
        assert len(self.all_dirs) == len(self.all_lables) # 断言，如果标签数和图片数不等退出程序
        
        indices = [i for i in range(len(self.all_dirs))]
        random.shuffle(indices) # 打乱all_lables和all_dirs的下表，方便接下来同时打乱这两个列表，不打乱对应关系
        
        self.all_dirs = [self.all_dirs[i] for i in indices]
        self.all_lables = [self.all_lables[i] for i in indices]
        
    def __len__(self): # 返回数据集的长度
        return len(self.all_lables)
    
    def __getitem__(self, idx):
        dir = self.all_dirs[idx]
        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE) # 读取路径对应的图片数据（灰度图的形式）  
        
        if img is None:
            raise ValueError(f"Image at index {idx} could not be loaded: {self.all_dirs[idx]}")
        
        img = img.astype(np.float32) / 255 # 归一化，把图片的值变到0~1范围
        img = np.expand_dims(img, 0)
        label = int(self.all_lables[idx])
        # cv2.imshow("image", img.squeeze(0))
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        
        return img, label
       


if __name__ == "__main__":
    dataset = MNIST_dataset("C:\\Users\\eason\\Desktop\\Learning Materials\\Third year\\MyPthonCourse\\myItem\\MNIST_Parsed\\MNIST_Parsed\\train")
    # print(dataset.all_dirs[0])
    # print(dataset.__len__())
    img, label = dataset.__getitem__(0)
# from torch import nn

# class MLP(nn.Module):
#     def __init__(self, nb_class):
#         super().__init__()
        
#         self.linear_relu_layers = nn.Sequential(
#             nn.Linear(28*28, 32), # 线性变换
#             nn.ReLU(), # 激活函数
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, nb_class)               
#         )
        
#         self.flatten = nn.Flatten()
    
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_layers(x)
#         return logits
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, nb_class):
        super(SimpleCNN, self).__init__()
        
        # # 卷积层和池化层
        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 输入通道1（灰度图），输出通道16
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),  # 使用2x2的池化窗口
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 输入通道16，输出通道32
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2)   # 再次池化
        # )
        
        # # 全连接层
        # self.fc_layers = nn.Sequential(
        #     nn.Flatten(),  # 展平卷积层输出以送入全连接层
        #     nn.Linear(32 * 7 * 7, 128),  # 根据卷积输出维度计算全连接层输入维度
        #     nn.ReLU(),
        #     nn.Linear(128, nb_class)  # 输出层
        # )
        # 卷积层和池化层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # 添加批量归一化
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # 添加批量归一化
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)  # 添加 Dropout
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 增加 Dropout
            nn.Linear(128, nb_class)
        )

    def forward(self, x):
    # 假设输入x的形状是 [batch_size, 784]
        x = x.view(-1, 1, 28, 28)  # 重塑输入为 [batch_size, channels, height, width]
        x = self.conv_layers(x)
        logits = self.fc_layers(x)
        return logits





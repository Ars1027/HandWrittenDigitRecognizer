import torch
import torch.nn.functional as F

class HandWrittenDigitRecognizer():
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.model = None
        
        self.load_model()
    
    def load_model(self):
        self.model = torch.load(self.model_path, map_location=torch.device('cpu')).to(self.device)
        # print(self.model)
        
        
    def predict(self, x):
        pred = self.model(x)
        # 使用 softmax 获取每个类别的预测概率
        prob = F.softmax(pred, dim=1)  # 对每一行（每个样本）应用 softmax
        y_hat = torch.argmax(pred, dim=1)
        
        return y_hat.item(), prob
    
    def predict_mult(self, x):
        pred = self.model(x)
        # 使用 softmax 获取每个类别的预测概率
        y_hat = torch.argmax(pred, dim=1)
        
        return y_hat.item()
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "C:\\Users\\eason\\Desktop\\Learning Materials\\Third year\\MyPthonCourse\\myItem\\model.pth"
        
    recognzier = HandWrittenDigitRecognizer(
        model_path=model_path,
        device=device,
    )
        
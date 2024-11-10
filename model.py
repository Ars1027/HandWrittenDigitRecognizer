import torch
from predict import HandWrittenDigitRecognizer

class Model():
    def __init__(self, args):
        self.args = args
        self.presenter = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.predictor = HandWrittenDigitRecognizer(
            model_path=self.args.model_path,
            device=self.device
        )
        
    def getLabelProb(self, x):
        return self.predictor.predict(x)
    
    def getLabel(self, x):
        return self.predictor.predict_mult(x)
    
    def setPresenter(self, presenter):
        self.presenter = presenter
        
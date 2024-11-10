import argparse

import sys
from PySide6.QtWidgets import QApplication
from model import Model
from view import View
from presenter import Presenter

def get_args_parser():
    
    parser = argparse.ArgumentParser()

    # 模型路径
    parser.add_argument(
        "-model_path", 
        dest = "model_path",
        default = "C:\\Users\\eason\\Desktop\\Learning Materials\\Third year\\MyPthonCourse\\myItem\\model.pth",
        type = str,
        help = "Path to the model."
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args_parser()
    app = QApplication(sys.argv)
    
    model = Model(args)
    view = View()
    presenter = Presenter(
        view=view, 
        model=model
    )
    
    view.show()
    app.exec()

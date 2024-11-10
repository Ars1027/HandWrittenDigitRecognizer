import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QProgressBar, QGroupBox
from PySide6.QtGui import QColor
from Canvas import Canvas

class View(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Handwriting Number Recognition")
        self.setFixedSize(800, 750)  # 增加宽度以容纳概率可视化

        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)
        
    def setPresenter(self, presenter):
        self.main_widget.setPresenter(presenter)
        
class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.presenter = None
        
        # 创建按钮布局
        btn_layout = QHBoxLayout()
        predict_btn = QPushButton(text="Predict individual num")
        predict_btn_2 = QPushButton(text="Predict multiple num")
        predict_btn_3 = QPushButton(text="Predict with Qwen")
        clear_btn = QPushButton(text="Clear")
        
        btn_layout.addWidget(predict_btn)
        btn_layout.addWidget(predict_btn_2)
        btn_layout.addWidget(predict_btn_3)
        btn_layout.addWidget(clear_btn)
        
        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)
        
        # 创建主画布和按钮的垂直布局
        left_layout = QVBoxLayout()
        self.canvas = Canvas()
        
        # 创建一个带白色背景的标签用于显示结果
        self.result_label = QLabel("Prediction result will appear here")
        self.result_label.setWordWrap(True)  # 启用自动换行
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 1px solid gray;
                padding: 10px;
                font-size: 14px;
            }
        """)

        # 将画布、按钮和标签添加到垂直布局
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(btn_widget)
        left_layout.addWidget(self.result_label)

        # 创建一个用于可视化概率的布局
        self.probability_layout = QVBoxLayout()
        self.progress_bars = []

        # 创建并添加 10 个 QProgressBar，分别用于数字 0 到 9
        for i in range(10):
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)  # 设置范围为 0 到 100（百分比）
            progress_bar.setTextVisible(True)
            progress_bar.setFormat(f"{i}: %p%")  # 显示数字标签
            self.probability_layout.addWidget(progress_bar)
            self.progress_bars.append(progress_bar)

        # 创建一个 QGroupBox 以包含进度条
        probability_box = QGroupBox("Prediction Probabilities")
        probability_box.setLayout(self.probability_layout)

        # 创建水平布局，将左侧的画布和按钮布局与右侧的概率可视化布局并排
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)      # 左侧布局
        main_layout.addWidget(probability_box)  # 右侧布局包含进度条

        # 设置主布局
        self.setLayout(main_layout)

        # 连接按钮点击信号
        clear_btn.clicked.connect(self.onClearBtnClicked)
        predict_btn.clicked.connect(self.onPredictBtnClicked)
        predict_btn_2.clicked.connect(self.onPredictBtnClicked_2)
        predict_btn_3.clicked.connect(self.onPredictBtnClicked_3)
        
    def onClearBtnClicked(self):
        self.canvas.clearCanvas()
        self.result_label.setText("Prediction result will appear here")  # 清空标签文本
        for progress_bar in self.progress_bars:
            progress_bar.setValue(0)  # 重置进度条
        
    def onPredictBtnClicked(self):
        if self.presenter is not None:
            self.presenter.onPredictBtnClicked(self.canvas.getImage(), self.onPrediction)  # 返回画布内容
        else:
            print("Presenter is none")
    
    def onPredictBtnClicked_2(self):
        if self.presenter is not None:
            self.presenter.onPredictMultipleBtnClicked(self.canvas.getImage(), self.onPrediction_2)  # 返回画布内容
    
    def onPredictBtnClicked_3(self):
        if self.presenter is not None:
            self.presenter.onPredictWithQwen(self.canvas.getImage_path(), self.onPrediction_3)  # 返回画布内容
        
    def setPresenter(self, presenter):
        self.presenter = presenter
        
    def onPrediction(self, label, prob):
        # print("prob shape:", prob.shape)  # 调试输出 prob 的形状
        result_text = f"Predicted result: {label}\n"
        print(result_text)
        self.result_label.setText(result_text)
        
        # 将每个数字的预测概率显示在进度条上
        for class_idx, prob_value in enumerate(prob[0]):  # 如果 prob 是 (batch_size, num_classes)，那么取第一个样本
            probability = prob_value.item()  # 获取第 i 个数字的概率
            self.progress_bars[class_idx].setValue(int(probability * 100))  # 将概率转换为百分比并更新进度条
            
    # 回调函数显示多个数字结果
    def onPrediction_2(self, results):
        if len(results) == 1:
            result_text = f"Predicted result: {''.join(str(result[0]) for result in results)}\n"
        else:
            result_text = f"Predicted result: {results}"
        print(result_text)
        self.result_label.setText(result_text)  # 将结果显示在标签上

        # 更新进度条 (假设 results 中包含每个数字的概率信息)
        if len(results) == 1 and isinstance(results[0][1], list):
            for i, p in enumerate(results[0][1]):
                self.progress_bars[i].setValue(int(p * 100))
                
    def onPrediction_3(self, results):
        self.result_label.setText("Predicted result:"+results)

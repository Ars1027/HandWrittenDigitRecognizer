import sys
import pyttsx3
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QProgressBar, QGroupBox
from PySide6.QtGui import QColor
from PySide6.QtCore import QTimer
from PySide6.QtCore import Qt
from camera import Camera
from Canvas import Canvas

class View(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Handwriting Number Recognition")
        # self.setFixedSize(850, 750)  
        self.setFixedSize(850, 850)  

        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)
        
    def setPresenter(self, presenter):
        self.main_widget.setPresenter(presenter)
        
class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.presenter = None
        
        
        # 初始化计时器
        self.inactivity_timer = QTimer()
        self.inactivity_timer.setInterval(1000)  # 每隔1秒触发一次，用于更新倒计时
        self.inactivity_timer.timeout.connect(self.updateCountdown)

        self.remaining_time = 5  # 设置倒计时初始时间为5秒

        # 创建倒计时标签
        self.countdown_label = QLabel("Auto prediction in: 5 seconds")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("font-size: 16px; color: red;")

        # 创建按钮布局
        btn_layout = QHBoxLayout()
        predict_btn = QPushButton(text="Predict individual num")
        predict_btn_2 = QPushButton(text="Predict multiple num")
        predict_btn_3 = QPushButton(text="Predict with Qwen")
        clear_btn = QPushButton(text="Clear")
        open_camera_btn = QPushButton(text="Open Camera") 

        btn_layout.addWidget(predict_btn)
        btn_layout.addWidget(predict_btn_2)
        btn_layout.addWidget(predict_btn_3)
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(open_camera_btn)

        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)

        # 创建主画布和按钮的垂直布局
        left_layout = QVBoxLayout()
        self.canvas = Canvas()
        self.canvas.input_detected.connect(self.onCanvasInput)  # 连接信号到事件处理

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

        # 将画布、按钮、倒计时标签和结果标签添加到垂直布局
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(btn_widget)
        left_layout.addWidget(self.result_label)
        left_layout.addWidget(self.countdown_label)

        # 可视化概率
        self.probability_layout = QVBoxLayout()
        self.progress_bars = []

        # 创建并添加 10 个 QProgressBar，分别用于数字 0 到 9
        for i in range(10):
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)  # 百分比
            progress_bar.setTextVisible(True)
            progress_bar.setFormat(f"{i}: %p%")  # 数字标签
            self.probability_layout.addWidget(progress_bar)
            self.progress_bars.append(progress_bar)

        # 创建一个 QGroupBox
        probability_box = QGroupBox("Prediction Probabilities")
        probability_box.setLayout(self.probability_layout)

        # 水平布局
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addWidget(probability_box)

        # 设置主布局
        self.setLayout(main_layout)

        # 连接按钮点击信号
        clear_btn.clicked.connect(self.onClearBtnClicked)
        predict_btn.clicked.connect(self.onPredictBtnClicked)
        predict_btn_2.clicked.connect(self.onPredictBtnClicked_2)
        predict_btn_3.clicked.connect(self.onPredictBtnClicked_3)
        open_camera_btn.clicked.connect(self.onOpenCameraBtnClicked)

        
    def onClearBtnClicked(self):
        self.canvas.clearCanvas()
        self.result_label.setText("Prediction result will appear here")  # 清空标签文本
        for progress_bar in self.progress_bars:
            progress_bar.setValue(0)  # 重置进度条

        self.inactivity_timer.stop()  # 停止计时器
        self.remaining_time = 5  # 重置倒计时时间
        self.updateCountdownLabel("Auto prediction in: 5 seconds")
        
    def onPredictBtnClicked(self):
        
        self.inactivity_timer.stop()  # 停止计时器
        self.remaining_time = 5  # 重置倒计时时间
        self.updateCountdownLabel("Auto prediction in: 5 seconds")
        
        if self.presenter is not None:
            self.presenter.onPredictBtnClicked(self.canvas.getImage(), self.onPrediction)  # 返回画布内容
                

    
    def onPredictBtnClicked_2(self):
        
        self.inactivity_timer.stop()  # 停止计时器
        self.remaining_time = 5  # 重置倒计时时间
        self.updateCountdownLabel("Auto prediction in: 5 seconds")
        
        if self.presenter is not None:
            self.presenter.onPredictMultipleBtnClicked(self.canvas.getImage(), self.onPrediction_2)  # 返回画布内容
    
    def onPredictBtnClicked_3(self):
        
        self.inactivity_timer.stop()  # 停止计时器
        self.remaining_time = 5  # 重置倒计时时间
        self.updateCountdownLabel("Auto prediction in: 5 seconds")
        
        if self.presenter is not None:
            self.presenter.onPredictWithQwen(self.canvas.getImage_path(), self.onPrediction_3)  # 返回画布内容
            
    def onOpenCameraBtnClicked(self):
        """打开摄像头"""
        self.camera = Camera(self.presenter)  # 创建 Camera 对象
        self.camera.video()  

        
    def setPresenter(self, presenter):
        self.presenter = presenter
        
    def onPrediction(self, label, prob):
        # print("prob shape:", prob.shape)  # 调试输出 prob 的形状
        result_text = f"识别结果是：{label}\n"
        print(result_text)
        self.result_label.setText(result_text)
        
        # 将每个数字的预测概率显示在进度条上
        for class_idx, prob_value in enumerate(prob[0]):  # 如果 prob 是 (batch_size, num_classes)，那么取第一个样本
            probability = prob_value.item()  # 获取第 i 个数字的概率
            self.progress_bars[class_idx].setValue(int(probability * 100))  # 将概率转换为百分比并更新进度条
        
        # 语音播报
        pt = pyttsx3.init()
        pt.say(result_text)
        pt.runAndWait()
               
    # 回调函数显示多个数字结果
    def onPrediction_2(self, results):
        if len(results) == 1:
            result_text = f"识别结果是：{''.join(str(result[0]) for result in results)}\n"
        else:
            result_text = f"识别结果是：{results}"
        print(result_text)
        self.result_label.setText(result_text)  # 将结果显示在标签上

        # 语音播报
        pt = pyttsx3.init()
        pt.say(result_text)
        pt.runAndWait()
                
    def onPrediction_3(self, results):
        self.result_label.setText(results)
        # 语音播报
        pt = pyttsx3.init()
        pt.say(results)
        pt.runAndWait()
        
    def updateCountdown(self):
        """每秒更新倒计时"""
        self.remaining_time -= 1
        if self.remaining_time > 0:
            self.updateCountdownLabel(f"Auto prediction in: {self.remaining_time} seconds")
        else:
            self.inactivity_timer.stop()
            self.autoPredict()

    def updateCountdownLabel(self, text):
        """更新倒计时标签内容"""
        self.countdown_label.setText(text)

    def autoPredict(self):
        """计时器超时后自动调用多数字预测"""
        self.onPredictBtnClicked_2()
        self.updateCountdownLabel("Prediction completed")

    def onCanvasInput(self):
        """当画布检测到输入时调用"""
        self.remaining_time = 5  # 重置倒计时
        self.updateCountdownLabel(f"Auto prediction in: {self.remaining_time} seconds")
        self.inactivity_timer.start()  # 每次输入都会重新启动计时器

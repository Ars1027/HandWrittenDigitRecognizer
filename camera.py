import cv2
import time
import numpy as np
from PySide6.QtGui import QImage

class Camera:
    def __init__(self, presenter):
        self.presenter = presenter
        self.ret = False
        self.frame = ''
        self.img = ''
        self.capture = None  # 保存 VideoCapture 对象的引用
        self.is_running = False  # 控制视频读取的标志

    def video(self):
        """启动摄像头"""
        self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 打开内置摄像头

        # 设置捕获帧的目标大小为 560×560
        target_size = (500, 500)

        while True:
            start_time = time.time()  # 记录起始时间

            self.ret, self.frame = self.capture.read()  # 读取摄像头帧

            if self.ret:
                # 调整帧大小为 560×560
                resized_frame = cv2.resize(self.frame, target_size, interpolation=cv2.INTER_AREA)

                # 将调整大小的帧转换为 QtGui.QImage
                qt_image = self.convert_frame_to_qimage(resized_frame)

                # 将当前帧传递给 Presenter 的多数字识别逻辑
                prediction_text = self.detect_digits(qt_image)

                # 在画面上叠加预测结果
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (0, 0, 255)  # 红色文字
                thickness = 2
                position = (10, 50)

                cv2.putText(resized_frame, prediction_text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

                # 添加退出提示文字
                exit_text = "Press 'q' to close camera"
                cv2.putText(resized_frame, exit_text, (10, 30), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

                cv2.imshow("Real-time Multi-Digit Recognition", resized_frame)

            # 限制帧率，提高性能
            elapsed_time = time.time() - start_time
            delay = max(1, int((1 / 10 - elapsed_time) * 1000))  # 目标帧率为10 FPS
            # 检测用户是否关闭窗口或者按下 'q' 键
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                self.capture.release()  # 释放摄像头
                cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
                break

    def detect_digits(self, qt_image):
        """实时数字检测"""
        final_result = []

        # 使用 Presenter 的方法进行数字识别
        # def callback_function(result):
        #     nonlocal final_result
        #     final_result = result
            
            
        # self.presenter.onPredictMultipleBtnClicked(qt_image, callback_function)
        
        
        def callback_function(label, prob):
            nonlocal final_result
            final_result = label

        self.presenter.onPredictBtnClicked(qt_image, callback_function)
        

        return f"predict result: {final_result}"

    def convert_frame_to_qimage(self, frame):
        """将 OpenCV 图像转换为 QtGui.QImage 格式"""
        # 检查帧是否为 RGB 图像，若为 BGR 则需要转换为 RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:  # 彩色图像
            height, width, channels = frame.shape
            bytes_per_line = channels * width
            # OpenCV 的默认格式为 BGR，需转换为 RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        elif len(frame.shape) == 2:  # 灰度图像
            height, width = frame.shape
            bytes_per_line = width
            qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            raise ValueError("Unsupported frame format for QImage conversion")

        return qt_image

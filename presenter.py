import numpy as np
import cv2
import torch
from LLM import Qwen

from PySide6 import QtGui

class Presenter():
    def __init__(self, view, model):
        self.view = view
        self.model = model
        
        self.view.setPresenter(self)
        self.model.setPresenter(self)
        
    def onPredictBtnClicked(self, img, callback_function):

        
        # 假设 img 是经过 _convertQImagetoMat() 转换的灰度图
        # 先把图像转换为灰度
        img = img.convertToFormat(QtGui.QImage.Format_Grayscale8)
        img = self._convertQImagetoMat(img)
        
        # cv2.imshow("1", img)
        # cv2.waitKey(0)

        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img[:, :, 0]

        # 获取图像大小
        h, w = img.shape[:2]

        # 去除白边，找到黑色内容的边界
        coords = cv2.findNonZero(255 - img)  # 找到黑色内容的坐标
        x, y, w, h = cv2.boundingRect(coords)  # 获取包含黑色内容的边界框

        # 裁剪图像到内容区域
        img_cropped = img[y:y+h, x:x+w]

        # 创建一个白底的 560x560 画布
        canvas_size = (560, 560)
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255

        # 获取裁剪后图像的大小
        h_cropped, w_cropped = img_cropped.shape[:2]

        # 计算粘贴位置以使图像内容居中
        y_offset = (canvas_size[0] - h_cropped) // 2
        x_offset = (canvas_size[1] - w_cropped) // 2

        # 将裁剪后的图像粘贴到画布的中心
        canvas[y_offset:y_offset + h_cropped, x_offset:x_offset + w_cropped] = img_cropped
        
        # cv2.imshow("1", canvas)
        # cv2.waitKey(0)

        # 把图片大小更改为模型可读取的大小（28x28）
        img = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA) 

        # 黑白颜色反转
        img = 255 - img 

        # 归一化
        img = img.astype(np.float32) / 255  # 归一化，把图片的值变到0~1范围
        # cv2.imshow("Centered Image", img)
        # cv2.waitKey(0)
               
        img = np.expand_dims(img, 0)
        
        img = torch.from_numpy(img)
        
        # 塞入神经网络
        label, prob = self.model.getLabelProb(img)
              
        callback_function(label, prob)
        
    
    def _convertQImagetoMat(self, img):
        
        width = img.width()
        height = img.height()
        depth = img.depth()
        
        ptr = img.constBits()
        img = np.array(ptr).reshape(height, width, depth//8)

        return img
    

# -----------------多数字识别--------------------------------------------
    def onPredictMultipleBtnClicked(self, img, callback_function_2):
        img = img.convertToFormat(QtGui.QImage.Format_Grayscale8)
        img = self._convertQImagetoMat(img)

        results = self.predict_multiple_chars(img)
        if len(results) == 1:
            final_result = results
        else:
            # 多个字符的情况，只显示标签序列        
            final_result = ''.join(str(result[0]) for result in results)
        callback_function_2(final_result)
    
    def preprocess_character(self, char_img, is_dot=False):
        # 获取字符图像的高度和宽度
        h, w = char_img.shape

        # 以长宽的最大值为边长创建正方形背景
        # side_length = max(h, w) 
        # square_img = np.zeros((side_length, side_length), dtype=np.uint8)
        square_img = np.ones((560, 560), dtype=np.uint8) * 255  # 白色背景

        # 计算字符图像在正方形背景中居中的起始坐标
        y_offset = (560 - h) // 2
        x_offset = (560 - w) // 2

        # 将字符图像复制到正方形背景中
        square_img[y_offset:y_offset + h, x_offset:x_offset + w] = char_img

        # 将正方形背景图像缩放为 28×28
        resized_img = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)

        # 反转颜色并归一化
        normalized_img = abs(255 - resized_img)  # 反转颜色
        # cv2.imshow('a',normalized_img)
        # cv2.waitKey(0)
        normalized_img = normalized_img.astype(np.float32) / 255  # 归一化
        normalized_img = np.expand_dims(normalized_img, 0)  # 添加通道维度
    
        return normalized_img


    def predict_multiple_chars(self, img):
        regions, min_areas_idx = self.digital_segmentation(img)
        results = []

        # 对每个分割的字符区域进行预处理和识别
        for idx, (region, bbox, area) in enumerate(regions):
            is_dot = (idx == min_areas_idx)  # 判断是否为小数点
            preprocessed_img = self.preprocess_character(region, is_dot=is_dot)
            
            # img_tensor = torch.from_numpy(preprocessed_img).unsqueeze(0)  # 添加批次维度
            img_tensor = torch.from_numpy(preprocessed_img)
            label = self.model.getLabel(img_tensor)
            # print(prob)
            # 如果是小数点区域，添加 '.'，否则添加识别到的数字标签
            if is_dot:
                results.append(('.', bbox))
            else:
                results.append((str(label), bbox))
            
        
        results.sort(key=lambda x: x[1][0])  # 根据x坐标排序
        return results
    
    def digital_segmentation(self, img):

        
        # 如果图像不是 numpy 数组，先将其转换为 numpy 数组
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # 如果图像是单通道格式 (height, width, 1)，将其转换为 (height, width)
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze(-1)  # 移除单通道的维度，变为 (height, width)

        # 确保图像是灰度或彩色格式
        if len(img.shape) == 3 and img.shape[2] == 3:
            # 图像是三通道彩色图像
            img = cv2.GaussianBlur(img, (3, 3), 0)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            # 图像是单通道灰度图
            gray = img
        else:
            raise ValueError("Unexpected image format. Image should be either a grayscale or BGR image.")

        # 应用中值滤波
        gray = cv2.medianBlur(gray, 5)

        # 二值化（已经将图片进行颜色反转了）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        # _, binary = cv2.threshold(gray, 0, 255,  cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        # cv2.imshow('a',binary)

        # 形态学操作，膨胀和闭运算
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernelX, iterations=3)
        kerne2X = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.dilate(binary, kerne2X)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        areas = []
        for item in contours:
            area = cv2.contourArea(item)
            x, y, w, h = cv2.boundingRect(item)
            regions.append((img[y:y + h, x:x + w], (x, y, w, h), area))
            areas.append((area, w, h))
            
        min_areas_idx = None
        max_area = max(areas, key=lambda x: x[0])[0]

        # 使用面积和宽高比判断小数点，先判断面积是否是最大面积的面积的一半及以下，再判断宽高比是否大于0.5
        potential_dots = [i for i, (area, w, h) in enumerate(areas) if area < max_area / 2 and min(w, h) / max(w, h) > 0.5]
        if potential_dots:
            min_areas_idx = potential_dots[0]
        

        # 返回分割后的图像区域和最小区域索引
        return regions, min_areas_idx
    
    
#---------------使用通义千问进行预测------------------
    def onPredictWithQwen(self, image_path, callback_function):
        LLM = Qwen(image_path)
        callback_function(LLM.outputResult())
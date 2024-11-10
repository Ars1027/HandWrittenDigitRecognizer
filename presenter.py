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
        # 把图片转换为灰度图
        img = img.convertToFormat(QtGui.QImage.Format_Grayscale8)
        # 图片信息转换为numpy矩阵信息
        img = self._convertQImagetoMat(img)
        # 把图片大小更改为模型可读取的大小（28x28）
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        # 黑白颜色反转
        img = 255 - img 
        #归一化
        img = img.astype(np.float32) / 255 # 归一化，把图片的值变到0~1范围
        
        # cv2.imshow("1", img)
        # cv2.waitKey(0)
        
        img = np.expand_dims(img, 0)
        
        img = torch.from_numpy(img)
        
        #  # 将图像展平
        # img = img.flatten()  # 将 28x28 转换为一维的 784

        # # 转换为 PyTorch 张量，并添加 batch 维度
        # img = torch.from_numpy(img).unsqueeze(0)  # 变成 (1, 784)
        # 塞入神经网络
        label, prob = self.model.getLabelProb(img)
        
        # print(f"Your input is label {label}")
        # print(prob)
        
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

    # def segment_image(self, img):
    #     # 应用阈值找到字符
    #     _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        
    #     # 寻找连通组件
    #     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        
    #     # 提取字符和小数点的区域
    #     regions = []
    #     for i in range(1, num_labels):  # 忽略背景
    #         x, y, w, h, area = stats[i]
    #         if area > 10:  # 忽略小的区域可能是噪声
    #             region = img[y:y+h, x:x+w]
    #             regions.append((region, (x, y, w, h)))
    #     return regions
    
    def segment_image(self, img):
        # 应用阈值找到字符
        _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        
        # 寻找连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        
        # 提取字符和小数点的区域
        regions = []
        max_area = 0  # 最大字符面积，用于判断小数点
        
        for i in range(1, num_labels):  # 忽略背景
            x, y, w, h, area = stats[i]
            max_area = max(max_area, area)  # 更新最大面积

            # 判断是否为小数点或字符
            if area > 10:  # 忽略小的区域可能是噪声
                region = img[y:y+h, x:x+w]
                regions.append((region, (x, y, w, h), area))
        
        # 过滤小数点的逻辑
        final_regions = []
        for region, bbox, area in regions:
            if area < max_area / 4:
                # 判定为小数点
                # 将小数点也加入regions，但不进行居中处理
                final_regions.append((region, bbox, 'dot'))
            else:
                # 判定为字符
                final_regions.append((region, bbox, 'char'))
        
        return final_regions

    # def preprocess_character(self, char_img):
    #     # 调整图像大小并归一化
    #     resized_img = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_AREA)
    #     resized_img = 255 - resized_img  # 反转颜色
    #     normalized_img = resized_img.astype(np.float32) / 255
    #     normalized_img = np.expand_dims(normalized_img, 0)  # 添加通道维度
    #     return normalized_img
    
    def preprocess_character(self, char_img, is_dot=False):
    # 确保图像为灰度格式
        if len(char_img.shape) == 3:
            if char_img.shape[2] == 3:  # 如果为三通道彩色图像
                char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
            elif char_img.shape[2] == 1:  # 单通道但形状为 (height, width, 1)
                char_img = char_img[:, :, 0]

        # 获取字符图像的高度和宽度
        h, w = char_img.shape

        # 以长宽的最大值为边长创建正方形背景
        side_length = max(h, w)
        square_img = np.zeros((side_length, side_length), dtype=np.uint8)

        # 计算字符图像在正方形背景中居中的起始坐标
        y_offset = (side_length - h) // 2
        x_offset = (side_length - w) // 2

        # 将字符图像复制到正方形背景中
        square_img[y_offset:y_offset + h, x_offset:x_offset + w] = char_img

        # 将正方形背景图像缩放为 28×28
        resized_img = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)

        # 反转颜色并归一化
        normalized_img = 255 - resized_img  # 反转颜色
        normalized_img = normalized_img.astype(np.float32) / 255  # 归一化
        normalized_img = np.expand_dims(normalized_img, 0)  # 添加通道维度
    
        return normalized_img





    def predict_multiple_chars(self, img):
        # regions = self.segment_image(img)
        # results = []
        
        # for region, bbox, region_type in regions:
        #     preprocessed_img = self.preprocess_character(region, is_dot=(region_type == 'dot'))
        #     img_tensor = torch.from_numpy(preprocessed_img).unsqueeze(0)  # 添加批次维度
        #     label, prob = self.model.getLabelProb(img_tensor)
        #     results.append((label, prob, bbox))
        regions, min_areas_idx = self.digital_segmentation(img)
        results = []

        # 对每个分割的字符区域进行预处理和识别
        for idx, (region, bbox, area) in enumerate(regions):
            is_dot = (idx == min_areas_idx)  # 判断是否为小数点
            preprocessed_img = self.preprocess_character(region, is_dot=is_dot)
            img_tensor = torch.from_numpy(preprocessed_img).unsqueeze(0)  # 添加批次维度
            label, prob = self.model.getLabelProb(img_tensor)
            # 如果是小数点区域，添加 `'.'`，否则添加识别到的数字标签
            if is_dot:
                results.append(('.', prob, bbox))
            else:
                results.append((str(label), prob, bbox))
        
        results.sort(key=lambda x: x[2][0])  # 根据x坐标排序
        return results
    
    def digital_segmentation(self, img):
        # 检查并打印图像的形状，以帮助调试
        print(f"Image shape before processing: {img.shape}")
        
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

        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

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
            # areas.append(area)
            areas.append((area, w, h))
            
        # # 判断小数点
        # min_areas_idx = None
        # if 5 * min(areas) < max(areas):
        #     min_areas_idx = areas.index(min(areas))
        min_areas_idx = None
        max_area = max(areas, key=lambda x: x[0])[0]

        # 使用面积和宽高比判断小数点
        potential_dots = [i for i, (area, w, h) in enumerate(areas) if area < max_area / 2 and min(w, h) / max(w, h) > 0.5]
        if potential_dots:
            min_areas_idx = potential_dots[0]
            print('小数点索引位置', min_areas_idx)
        else:
            print('无小数点')

        # 返回分割后的图像区域和最小区域索引
        return regions, min_areas_idx
    
    
#---------------使用通义千问进行预测------------------
    def onPredictWithQwen(self, image_path, callback_function):
        LLM = Qwen(image_path)
        callback_function(LLM.outputResult())
        
        


    
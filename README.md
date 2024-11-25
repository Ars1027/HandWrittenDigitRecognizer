# HandWrittenDigitRecognizer
手写数字识别：一个关于AI界helloworld的项目(
# 环境依赖
+ python==3.10
+ numpy==1.22.4
+ opencv_python==4.6.0.66
+ PySide6==6.4.0.1
+ PySide6_Addons==6.4.0.1
+ PySide6_Essentials==6.4.0.1
+ torch==1.11.0
+ tqdm==4.61.1
+ openai==1.54.3

# 更新日志
+ 2024/11/14 提高识别精度
+ 2024/11/18 增加语音播报以及无操作后5s自动识别
+ 2024/11/21 可以使用摄像头识别（精度有待提高）


## 使用了MVP架构（model，view，presenter）
分别为model.py，view.py(pyqt实现)，presenter.py
## 模型训练结果
model.pth是已经训练好的模型，若不想自行训练可直接使用
测试集上98%的准确率
![image](https://github.com/Ars1027/HandWrittenDigitRecognizer/blob/main/results/%E6%B5%8B%E8%AF%95%E9%9B%86%E7%BB%93%E6%9E%9C.png)
## 识别
四种方式
### 0-9单数字识别：给出预测结果以及神经网络预测各个数字的概率
![image](https://github.com/Ars1027/HandWrittenDigitRecognizer/blob/main/results/%E5%8D%95%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E7%BB%93%E6%9E%9C.png)
### 多数字识别（含小数点）
**_实现思路：使用opencv把画板上的数字分割成一个个矩形，再将其重新扩张为一张28x28的图片供模型预测。小数点的识别逻辑是：若发现了一个矩形的面积明显的小于其他矩形的面积，则说明是小数点_**
![image](https://github.com/Ars1027/HandWrittenDigitRecognizer/blob/main/results/%E5%A4%9A%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB(%E5%B8%A6%E5%B0%8F%E6%95%B0%E7%82%B9).png)
### 调用大语言模型进行预测（这里使用的是阿里的通义千问大模型，具体配置方法可前往此处进行https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api）
![image](https://github.com/Ars1027/HandWrittenDigitRecognizer/blob/main/results/%E4%BD%BF%E7%94%A8%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%AF%86%E5%88%AB.png)
### 摄像头实时识别

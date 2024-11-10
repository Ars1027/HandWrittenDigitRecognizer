from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6 import QtGui
from PySide6.QtWidgets import QMessageBox
from PIL import ImageTk, Image

class Canvas(QLabel):
    def __init__(self):
        super().__init__()
        
        self.canvas = QtGui.QPixmap(560, 560)
        self.canvas.fill(Qt.white)
        # self.canvas.fill(Qt.black)
        self.setPixmap(self.canvas)
        
        self.last_x, self.last_y = None, None # 记录上一个点的坐标
        
    # 重写鼠标点击事件函数
    def mouseMoveEvent(self, ev):
        # print(ev.x(), ev.y())
        if self.last_x is None or self.last_y is None:
            self.last_x = ev.x()
            self.last_y = ev.y()
            return
        
        painter = QtGui.QPainter(self.canvas)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)  # 启用抗锯齿
        # 设置画笔
        p = painter.pen()
        # p.setColor(Qt.white)
        p.setWidth(20)
        p.setCapStyle(Qt.RoundCap)
        p.setJoinStyle(Qt.RoundJoin)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, ev.x(), ev.y())
        
        # 使用贝塞斯曲线绘制
        # path = QtGui.QPainterPath()
        # path.moveTo(self.last_x, self.last_y)
        # path.cubicTo((self.last_x + ev.x()) / 2, self.last_y, (self.last_x + ev.x()) / 2, ev.y(), ev.x(), ev.y())
        # painter.drawPath(path)
        
        # 更新坐标信息
        self.last_x = ev.x()
        self.last_y = ev.y()
        
        painter.end()
        # 刷新
        self.update()
        
    # 重写鼠标释放事件函数         
    def mouseReleaseEvent(self, ev):
        self.last_x, self.last_y = None, None       
    
    # 重写绘画事件函数
    def paintEvent(self, arg__1):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0, 0, self.canvas)   
        
    # 清除Canvas成功信息弹窗
    def onClearCompleted(self):
        QMessageBox.information(None, "inmformation", "Canvas has been cleared succesfully!")    
   
    # 清空Canvas
    def clearCanvas(self):
        self.canvas.fill(Qt.white) #将画布重新填充为白色
        # self.canvas.fill(Qt.black) #将画布重新填充为黑色
        self.update()
        self.onClearCompleted()
    
    # 返回画布内容，转换成图片
    def getImage(self):
        return self.canvas.toImage()
    
    def getImage_path(self):
        img = self.canvas.toImage()
        # 保存图片到本地
        img.save("canvas_image.png")
        return "canvas_image.png"
    
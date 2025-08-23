import os

from PySide6 import QtCore
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QImage, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QFileDialog


class MImageViewer(QWidget):
    # 自定义信号
    imageDragged = Signal(QPointF)
    imageScaled = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)  # 设置最小尺寸确保有足够显示空间
        self.setMouseTracking(True)

        # 图像相关
        self.original_image = None
        self.display_image = None
        self.result_image = None        #Qimage
        self.curr_result_array  = None
        self.current_scale = 1.0
        self.drag_start_pos = None
        self.status = "IDLE"
        self.image_pos = QPointF(0, 0)

        self.show_original = True

        # 显示设置
        self.pens = []
        for color in [QColor(0, 255, 0,255),QColor(255, 0, 0,255)]:
            pen = QPen()
            pen.setColor(color)
            brush = QBrush(color)
            pen.setBrush(brush)
            pen.setWidth(2)
            self.pens.append(pen)
        self.point_radius = 5

    def trans_img(self, img_array):
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        if ch == 1:
            return QImage(img_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        elif ch == 3:
            return QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        elif ch == 4:
            return QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        raise ValueError(f"Invalid image channel:{ch}")

    def set_image(self, cv_img):
        self.original_image = self.trans_img(cv_img)
        self.display_image = self.original_image
        self._fit_image_to_view()
        self.update()
        self.repaint()

    def set_result_image(self, result):
        self.curr_result_array = result
        self.result_image = self.trans_img(result)
        self.update()

    def get_result_array(self):
        return self.curr_result_array

    def paintEvent(self, event):
        if self.display_image is None:
            return
        painter = QPainter(self)

        sx = self.original_image.width() / self.display_image.width()
        sy = self.original_image.width() / self.display_image.height()
        # 计算缩放后的图像尺寸和位置
        scaled_width = int(sx * self.display_image.width() * self.current_scale)
        scaled_height = int(sy * self.display_image.height() * self.current_scale)
        x = int(self.image_pos.x())
        y = int(self.image_pos.y())

        # 绘制图像
        painter.drawImage(x,y,
                          self.display_image.scaled(scaled_width, scaled_height,
                                                    Qt.KeepAspectRatio, Qt.SmoothTransformation))

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if self.display_image is None:
            return

        scaled_width = int(self.display_image.width() * self.current_scale)
        scaled_height = int(self.display_image.height() * self.current_scale)

        # 计算点击位置相对于图像的坐标 (0-1)
        img_x = (event.position().x() - self.image_pos.x()) / scaled_width
        img_y = (event.position().y() - self.image_pos.y()) / scaled_height

        # 检查点击是否在图像范围内
        if 0 <= img_x <= 1 and 0 <= img_y <= 1:
            if event.button() == Qt.LeftButton:
                self.status = "LPRESS"

            elif event.button() == Qt.RightButton:
                self.status = "RPRESS"

        # 记录拖动起始位置
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drag_start_pos is not None and self.display_image is not None:
            # 计算移动距离
            self.status = "DRAG"
            delta = event.position() - self.drag_start_pos
            self.image_pos += delta
            self.drag_start_pos = event.position()
            self.imageDragged.emit(self.image_pos)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = None

        if self.display_image is None:
            return
        if self.status == "DRAG":
            self.status = "IDLE"
            return

    def wheelEvent(self, event: QWheelEvent):
        if self.display_image is None:
            return

        # 获取鼠标位置相对于图像的位置
        mouse_pos = event.position()
        old_img_x = (mouse_pos.x() - self.image_pos.x()) / (self.display_image.width() * self.current_scale)
        old_img_y = (mouse_pos.y() - self.image_pos.y()) / (self.display_image.height() * self.current_scale)

        # 计算缩放因子
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.current_scale *= zoom_factor
        self.current_scale = max(0.1, min(20.0, self.current_scale))  # 限制缩放范围

        # 调整图像位置，使鼠标下的点保持不动
        new_img_width = self.display_image.width() * self.current_scale
        new_img_height = self.display_image.height() * self.current_scale

        self.image_pos.setX(mouse_pos.x() - old_img_x * new_img_width)
        self.image_pos.setY(mouse_pos.y() - old_img_y * new_img_height)

        self.imageScaled.emit(self.current_scale)
        self.update()

    def toggle_image_display(self):
        if self.result_image is not None:
            self.show_original = not self.show_original
            self.display_image = self.original_image if self.show_original else self.result_image
            self.update()

    def show_result(self):
        if self.result_image is not None:
            self.show_original = False
            self.display_image = self.result_image
            self.update()

    def _fit_image_to_view(self):
        """调整图像大小和位置以适配视图"""
        if self.original_image is None:
            return

        # 获取视图和图像的尺寸
        view_width = self.width()
        view_height = self.height()
        img_width = self.original_image.width()
        img_height = self.original_image.height()

        # 计算保持宽高比的缩放比例
        width_ratio = view_width / img_width
        height_ratio = view_height / img_height
        self.current_scale = min(width_ratio, height_ratio)

        # 计算居中位置
        scaled_width = img_width * self.current_scale
        scaled_height = img_height * self.current_scale
        self.image_pos = QPointF(
            (view_width - scaled_width) / 2,
            (view_height - scaled_height) / 2
        )

    def resizeEvent(self, event):
        """窗口大小改变时重新调整图像"""
        self._fit_image_to_view()
        super().resizeEvent(event)


class TitledLineEdit(QWidget):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)

        # 子控件：QLabel 和 QLineEdit
        self.label = QLabel(title)
        self.edit = QLineEdit()

        # 设置布局（水平放置）
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    # 兼容 QLineEdit 的常用接口
    def text(self):
        return self.edit.text()

    def setText(self, text: str):
        self.edit.setText(text)

    def setValidator(self, validator):
        self.edit.setValidator(validator)

    def setPlaceholderText(self, text: str):
        self.edit.setPlaceholderText(text)

    def placeholderText(self):
        return self.edit.placeholderText()

    def setTitle(self, title: str):
        self.label.setText(title)

    def title(self):
        return self.label.text()

    def setReadOnly(self, ro: bool):
        self.edit.setReadOnly(ro)

    def isReadOnly(self):
        return self.edit.isReadOnly()

    def lineEdit(self):
        """返回内部的 QLineEdit 对象，便于高级操作"""
        return self.edit


def createPannel(widgets, parent=None, horizontal=True):
    pannel = QWidget(parent)
    if horizontal:
        layout = QHBoxLayout()
    else:
        layout = QVBoxLayout()
    for w in widgets:
        layout.addWidget(w)
    pannel.setLayout(layout)
    return pannel


def getFileSelectDialog(parent=None, showRoot=True):
    dialog = QFileDialog(parent)
    dialog.setWindowTitle("选择文件或目录")
    dialog.setFileMode(QFileDialog.ExistingFiles)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # 使用Qt对话框
    dialog.setOption(QFileDialog.ShowDirsOnly, False)
    dialog.setOption(QFileDialog.HideNameFilterDetails, False)
    # 添加自定义控件
    # dialog.setOption(QFileDialog.DontResolveSymlinks)
    if showRoot:
        home = os.path.expanduser("~")
        files = ["/", home]
        for f in os.listdir(home):
            fpath = os.path.join(home, f)
            files.append(fpath)
        dialog.setSidebarUrls([
            QtCore.QUrl.fromLocalFile(_) for _ in files
        ])  # 添加侧边栏
    else:
        dialog.setSidebarUrls([])
    for child in dialog.findChildren(QLineEdit):
        if child.placeholderText().lower() in ["file name", "文件名"]:
            # 找到文件名输入框旁边的父控件
            parent = child.parent()
            if parent:
                # 创建路径输入框
                path_edit = QLineEdit(parent)
                path_edit.setPlaceholderText("或直接输入完整路径")
                parent.layout().insertWidget(0, path_edit)

                # 连接信号
                def update_path(text):
                    if "/" in text or "\\" in text:  # 检测路径输入
                        child.setText(text)

                path_edit.textChanged.connect(update_path)
            break
    return dialog





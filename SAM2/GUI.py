import os
import sys
import traceback

import cv2
import numpy as np
from PySide6 import QtCore
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QScrollArea, QFrame, QComboBox, QCheckBox, QTableView,
                               QStyledItemDelegate, QProgressDialog, QListView, QSplitter, QDialog, QDialogButtonBox)
from PySide6.QtCore import Qt, QPoint, QPointF, Signal, QModelIndex
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent, QWheelEvent, QStandardItemModel, \
    QStandardItem, QBrush
import logging

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PySide6.QtWidgets import QFileDialog

palette = np.array([
    [0, 0, 0],
    [255, 0, 0],        # 红
    [255, 128, 0],      # 橙
    [255, 255, 0],      # 黄
    [128, 255, 0],      # 黄绿
    [0, 255, 0],        # 绿
    [0, 255, 128],      # 青绿
    [0, 255, 255],      # 青
    [0, 128, 255],      # 天蓝
    [0, 0, 255],        # 蓝
    [128, 0, 255],      # 紫
    [255, 0, 255],      # 品红
    [255, 0, 128],      # 玫瑰红
    [128, 64, 0],       # 深棕
    [0, 128, 64],       # 松绿
    [64, 0, 128],       # 靛蓝
    [255, 128, 128],    # 浅红
    [128, 255, 128],    # 浅绿
    [128, 128, 255],    # 浅蓝
    [255, 165, 0],      # 橙黄
    [255, 69, 0],       # 橙红
    [50, 205, 50],      # 酸橙绿
    [30, 144, 255],     # 道奇蓝
    [138, 43, 226],     # 蓝紫
    [147, 112, 219],    # 中紫
    [199, 21, 133],     # 深粉
    [255, 105, 180],    # 热粉
    [0, 199, 140],      # 绿松石
    [70, 130, 180],     # 钢蓝
    [123, 104, 238],    # 中蓝紫
    [240, 128, 128],    # 亮珊瑚
    [154, 205, 50],     # 黄绿
    [205, 92, 92],      # 印度红
    [255, 160, 122],    # 浅橙
    [218, 165, 32],     # 金
    [189, 183, 107],    # 暗卡其
    [107, 142, 35],     # 橄榄绿
    [0, 128, 128],      # 深青
    [72, 61, 139],      # 暗蓝紫
    [148, 0, 211],      # 暗紫
    [139, 0, 0],        # 暗红
    [233, 150, 122],    # 深橙
    [0, 206, 209],      # 深绿松石
    [95, 158, 160],     # 卡其蓝
    [100, 149, 237],    # 矢车菊蓝
    [210, 105, 30],     # 巧克力色
    [85, 107, 47],      # 暗橄榄绿
    [255, 215, 0],      # 金
    [218, 112, 214],    # 兰紫
    [176, 48, 96],      # 深玫瑰
    [102, 205, 170]     # 中海蓝
], dtype=np.uint8)

class PointWithPrompt:
    idx = 0
    def __init__(self, pos, prompt="foreground"):
        self.position = pos  # QPointF (相对坐标，0-1)
        # 提示词输入框
        prompt_input = QComboBox()
        prompt_input.addItems(["background", "foreground"])
        prompt_input.setCurrentText(prompt)
        self.prompt = prompt_input
        self.id = PointWithPrompt.idx #id(self)  # 唯一标识
        PointWithPrompt.idx += 1


class ComboBoxDelegate(QStyledItemDelegate):
    indexChanged = Signal()
    def __init__(self, items, default="", parent=None):
        super().__init__(parent)
        self.items = items
        self.default = default

    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        editor.addItems(self.items)
        if self.default in self.items:
            editor.setCurrentText(self.default)
        editor.currentIndexChanged.connect(self.somIdxChanged)
        return editor

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.EditRole)
        editor.setCurrentText(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def somIdxChanged(self):
        self.indexChanged.emit()

class ImageViewer(QWidget):
    # 自定义信号
    pointAdded = Signal(PointWithPrompt)
    pointRemoved = Signal(int)  # 传递point id
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
        self.result_array = None        #all layers
        self.curr_result_array  = None
        self.current_scale = 1.0
        self.drag_start_pos = None
        self.status = "IDLE"
        self.image_pos = QPointF(0, 0)

        # 关键点
        self.points = []
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
        # self.point_color = [QColor(0, 255, 0,255), QColor(255, 0, 0,255)]
        # self.pen = QPen(self.point_color)
        # self.pen.setWidth(2)

    def set_image(self, cv_img):
        # 读取图像
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w

        self.points = []
        PointWithPrompt.idx = 0
        self.original_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.display_image = self.original_image
        self._fit_image_to_view()
        self.update()
        self.repaint()

    def set_result_image(self, result, channels2show=None):
        # 设置分割结果
        self.result_array = result
        self.curr_result_array = None
        self.switch_result_layer(channels2show)
        if not self.show_original:
            self.update()

    def switch_result_layer(self, channels):
        if self.result_array is None:
            return
        if channels is None or len(channels) == 0:
            self.result_image = None
            if not self.show_original:
                self.display_image = self.original_image
        else:
            h,w = self.result_array.shape[:2]
            curr_result_array = np.zeros((h,w,4),dtype=np.uint8)
            for c in channels:
                curr_result_array[..., c] = self.result_array[..., c]
            curr_result_array[...,3]=255
            bytes_per_line = 4 * w
            self.curr_result_array = curr_result_array
            result = QImage(self.curr_result_array, w, h, bytes_per_line, QImage.Format_RGBA8888)
            self.result_image = result
        self.update()

    def get_result_array(self):
        return self.curr_result_array

    def paintEvent(self, event):
        if self.display_image is None:
            return
        painter = QPainter(self)

        # 计算缩放后的图像尺寸和位置
        scaled_width = int(self.display_image.width() * self.current_scale)
        scaled_height = int(self.display_image.height() * self.current_scale)

        # 绘制图像
        painter.drawImage(self.image_pos.x(), self.image_pos.y(),
                          self.display_image.scaled(scaled_width, scaled_height,
                                                    Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 绘制关键点
        if self.show_original:
            # self.pen.setColor(self.point_color)
            # painter.setPen(self.pen)

            for point in self.points:
                # 将相对坐标转换为实际坐标
                idx = point.prompt.currentIndex()
                painter.setPen(self.pens[idx])
                x = self.image_pos.x() + point.position.x() * scaled_width
                y = self.image_pos.y() + point.position.y() * scaled_height

                painter.drawEllipse(QPointF(x, y), self.point_radius, self.point_radius)
                # 可选: 显示点ID
                # painter.drawText(QPointF(x + 10, y), str(point.id))
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
                # 左键添加点
                # new_point = PointWithPrompt(QPointF(img_x, img_y))
                # self.points.append(new_point)
                # self.pointAdded.emit(new_point)
                # self.update()

            elif event.button() == Qt.RightButton:
                # 右键删除点: 找到最近的点
                if not self.points:
                    return
                self.status = "RPRESS"
                # 计算所有点的屏幕距离
                # distances = []
                # for point in self.points:
                #     px = self.image_pos.x() + point.position.x() * scaled_width
                #     py = self.image_pos.y() + point.position.y() * scaled_height
                #     dist = (px - event.position().x()) ** 2 + (py - event.position().y()) ** 2
                #     distances.append(dist)

                # 找到最近的点
                # min_idx = np.argmin(distances)
                # if distances[min_idx] < (self.point_radius * 2) ** 2:  # 阈值
                #     removed_id = self.points[min_idx].id
                #     del self.points[min_idx]
                #     self.pointRemoved.emit(removed_id)
                #     self.update()

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
        self.status = "IDLE"
        scaled_width = int(self.display_image.width() * self.current_scale)
        scaled_height = int(self.display_image.height() * self.current_scale)

        # 计算点击位置相对于图像的坐标 (0-1)
        img_x = (event.position().x() - self.image_pos.x()) / scaled_width
        img_y = (event.position().y() - self.image_pos.y()) / scaled_height
        if 0 <= img_x <= 1 and 0 <= img_y <= 1:
            if event.button() == Qt.LeftButton:
                # 左键添加点
                new_point = PointWithPrompt(QPointF(img_x, img_y))
                self.points.append(new_point)
                self.pointAdded.emit(new_point)
                self.update()

            elif event.button() == Qt.RightButton:
                # 右键删除点: 找到最近的点
                if not self.points:
                    return
                # 计算所有点的屏幕距离
                distances = []
                for point in self.points:
                    px = self.image_pos.x() + point.position.x() * scaled_width
                    py = self.image_pos.y() + point.position.y() * scaled_height
                    dist = (px - event.position().x()) ** 2 + (py - event.position().y()) ** 2
                    distances.append(dist)

                # 找到最近的点
                min_idx = np.argmin(distances)
                if distances[min_idx] < (self.point_radius * 2) ** 2:  # 阈值
                    removed_id = self.points[min_idx].id
                    del self.points[min_idx]
                    self.pointRemoved.emit(removed_id)
                    self.update()

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
        self.current_scale = max(0.1, min(5.0, self.current_scale))  # 限制缩放范围

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


class PointsPanel(QWidget):
    pointLabelChanged = Signal()
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.title = QLabel("关键点和")
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(self.title)

        self.point_table_view = QTableView()
        self.point_table_model = QStandardItemModel()
        self.point_table_view.setModel(self.point_table_model)
        self.point_table_model.setColumnCount(3)
        self.point_table_model.setHeaderData(0, Qt.Horizontal, "ID")
        self.point_table_model.setHeaderData(1, Qt.Horizontal, "点")
        self.point_table_model.setHeaderData(2, Qt.Horizontal, "标签")
        self.layout.addWidget(self.point_table_view)
        self.point_table_view.setFocusPolicy(Qt.NoFocus)
        combo_delegate = ComboBoxDelegate(["background", "foreground"], "foreground", self)
        combo_delegate.indexChanged.connect(self.somePointChanged)
        self.point_table_view.setItemDelegateForColumn(2,combo_delegate)
        # 添加一些间距
        self.layout.addStretch()

    def add_point(self, point: PointWithPrompt):
        self.point_table_model.appendRow([
            QStandardItem(str(point.id)),
            QStandardItem(f"({point.position.x():.2f},{point.position.y():.2f}"),
            QStandardItem("foreground")])

    def remove_point(self, point_id):
        # 查找并移除对应的点控件
        for i in range(self.point_table_model.rowCount(QModelIndex())):
            item = self.point_table_model.item(i, 0)
            if item is None:
                continue
            item = item.text()
            if item == str(point_id):
                self.point_table_model.removeRow(i)
                break
    def clean(self):
        self.point_table_model.clear()

    def somePointChanged(self):
        self.pointLabelChanged.emit()

class SAM2GUI(QMainWindow):
    def __init__(self,
                 checkpoint="./checkpoints/sam2.1_hiera_large.pt",
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                 ):
        super().__init__()
        self.setWindowTitle("SAM2 图像分割工具")

        # 主窗口布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.file_view = QListView(parent=self)
        self.file_item_model = QStandardItemModel()
        self.file_view.setModel(self.file_item_model)
        self.file_view.selectionModel().selectionChanged.connect(self.load_image)
        self.file_view.setFocusPolicy(Qt.NoFocus)
        self.image_files = []
        main_layout.addWidget(self.file_view, stretch=1)

        # 图像显示区域
        self.image_viewer = ImageViewer()
        self.image_viewer.setFocusPolicy(Qt.NoFocus)
        main_layout.addWidget(self.image_viewer, stretch=3)

        # 右侧控制面板
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(10, 10, 10, 10)

        # 关键点和提示词面板
        self.points_panel = PointsPanel()
        self.points_panel.pointLabelChanged.connect(self.image_viewer.update)
        self.points_panel.setFocusPolicy(Qt.NoFocus)
        right_panel.addWidget(self.points_panel)

        # 操作按钮
        self.load_button = QPushButton("加载图像")
        self.load_button.setFocusPolicy(Qt.NoFocus)
        self.segment_button = QPushButton("执行分割")
        self.segment_button.setFocusPolicy(Qt.NoFocus)
        self.clear_button = QPushButton("清除所有点")
        self.clear_button.setFocusPolicy(Qt.NoFocus)
        self.save_button = QPushButton("保存")
        self.save_button.setFocusPolicy(Qt.NoFocus)
        self.save_cfg_button = QPushButton("保存路径")
        self.save_cfg_button.setFocusPolicy(Qt.NoFocus)
        layout = QHBoxLayout()
        layout.addWidget(self.save_button)
        layout.addWidget(self.save_cfg_button)
        save_pannel = QWidget(self)
        save_pannel.setLayout(layout)
        self.multi_seg = QCheckBox('多对象分割')
        self.multi_seg.setFocusPolicy(Qt.NoFocus)
        self.multi_seg.setChecked(True)

        self.result_table_view = QTableView()
        self.result_table_model = QStandardItemModel()
        self.result_table_view.setModel(self.result_table_model)
        self.result_table_model.setColumnCount(3)
        self.result_table_model.setHeaderData(0, Qt.Horizontal, "ID")
        self.result_table_model.setHeaderData(1, Qt.Horizontal, "置信度")
        self.result_table_view.setFocusPolicy(Qt.NoFocus)
        self.result_table_view.setSelectionBehavior(QTableView.SelectRows)  # 关键设置
        # 设置选择模式（单选或多选）
        self.result_table_view.setSelectionMode(QTableView.SingleSelection)  # 单选一行
        # 或者
        # self.result_table_view.setSelectionMode(QTableView.MultiSelection)  # 多选行
        self.result_table_model.itemChanged.connect(self.change_result_layer)   #监听checkbox状态

        colored_label = QLabel()
        colored_label.setTextFormat(Qt.RichText)  # 允许富文本
        self.status_label = colored_label
        # 添加到状态栏的永久区域
        self.statusBar().addPermanentWidget(colored_label)

        right_panel.addWidget(self.load_button)
        right_panel.addWidget(self.segment_button)
        right_panel.addWidget(self.clear_button)
        right_panel.addWidget(save_pannel)
        right_panel.addWidget(self.multi_seg)
        right_panel.addWidget(self.result_table_view)
        right_panel.addStretch()

        main_layout.addLayout(right_panel, stretch=1)

        # 连接信号
        self.image_viewer.pointAdded.connect(self.points_panel.add_point)
        self.image_viewer.pointRemoved.connect(self.points_panel.remove_point)
        self.load_button.clicked.connect(self.load_files)
        self.segment_button.clicked.connect(self.perform_segmentation)
        self.clear_button.clicked.connect(self.clear_all_points)
        self.save_cfg_button.clicked.connect(self.set_save_dir)
        self.save_button.clicked.connect(self.save_result)

        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        self.curr_file = None
        self.curr_img = None
        self.curr_result = None
        self.save_dir = None
        self.layer_to_show = []
        self.setGeometry(100, 100, 1200, 800)

    def load_files(self):
        # 设置文件对话框选项
        dialog = QFileDialog(self)
        dialog.setWindowTitle("选择文件或目录")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # 使用Qt对话框

        # 添加自定义控件
        dialog.setOption(QFileDialog.DontResolveSymlinks)
        dialog.setSidebarUrls([QtCore.QUrl.fromLocalFile("/")])  # 添加侧边栏

        if dialog.exec():
            selected = dialog.selectedFiles()
            self.file_item_model.clear()
            self.image_files = []
            for file in selected:
                s = os.path.splitext(file)[-1].lower()
                if s not in [".jpg",".jpeg",".png",".tiff"]:
                    continue
                self.image_files.append({"src":file,"result":""})
                self.file_item_model.appendRow(QStandardItem(os.path.basename(file)))
            if len(self.image_files) > 0:
                self.file_view.setCurrentIndex(self.file_item_model.index(0,0))
        self.show_status()

    def load_image(self, item):
        idx = self.file_view.currentIndex().row()
        file_path = self.image_files[idx]["src"]
        print(file_path)
        # 如果用户选择了文件（没有取消对话框）
        if file_path:
            # 设置图像并重置视图状态
            self.curr_file = file_path
            cv_img = cv2.imread(file_path)
            self.curr_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            self.image_viewer.current_scale = 1.0  # 重置缩放
            self.image_viewer.image_pos = QPointF(0, 0)  # 重置位置
            self.image_viewer.set_image(self.curr_img)
            # self.clear_all_points()  # 清除之前的点

            # 可选：更新窗口标题显示当前文件名
            fname = os.path.splitext(file_path)[0]
            self.setWindowTitle(f"SAM2 图像分割工具 - {fname}")

    def perform_segmentation(self):
        # 这里应该调用SAM2的分割功能
        self.result_table_model.clear()
        logging.info("执行分割操作...")

        # 准备输入数据
        h, w = self.curr_img.shape[:2]
        points = np.array([(p.position.x()*w, p.position.y()*h) for p in self.image_viewer.points])
        labels = np.array([p.prompt.currentIndex() for p in self.image_viewer.points])

        # 调用SAM2模型进行分割
        multi_seg = self.multi_seg.isChecked()
        self.predictor.set_image(self.curr_img)
        #logits可用于下一次迭代
        progress = QProgressDialog("正在执行分割操作...", "取消", 0, 0, self)
        progress.setWindowTitle("请稍候")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)  # 移除取消按钮
        progress.setRange(0, 0)  # 设置为不确定进度模式（环形动画）
        progress.show()

        # 强制立即显示对话框
        QApplication.processEvents()

        try:
            self.curr_result = None
            masks, scores, logits= self.predictor.predict(
                points,
                labels,
                mask_input=None,         #可用于下一次迭代
                multimask_output=multi_seg
            )
            logging.info("分割操作完成")
            c, h, w = masks.shape

            result = np.clip(masks * 255, 0, 255)
            result = result.astype(np.uint8)
            if multi_seg:
                if len(self.layer_to_show) == 0:
                    self.layer_to_show = list(range(3))
                idx = np.argsort(scores)[::-1]  #概率从高向低排列
                result = result[idx]
                result = np.transpose(result, (1,2,0)).squeeze()
                for i in range(3):
                    id = idx[i]
                    score_item = QStandardItem(f"{scores[id]*100:.2f}%")
                    id_item = QStandardItem(str(i))
                    id_item.setCheckable(True)
                    if id in self.layer_to_show:
                        id_item.setCheckState(Qt.Checked)
                    else:
                        id_item.setCheckState(Qt.Unchecked)
                    self.result_table_model.appendRow([id_item, score_item])
            else:
                if len(self.layer_to_show) == 0:
                    self.layer_to_show = [0]
                score_item = QStandardItem(f"{scores[0] * 100:.2f}%")
                id_item = QStandardItem("0")
                id_item.setCheckable(True)
                id_item.setCheckState(Qt.Checked)
                self.result_table_model.appendRow([id_item, score_item])
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGBA)
            self.curr_result = result
            self.image_viewer.set_result_image(result, self.layer_to_show)
        except Exception:
            logging.error(traceback.format_exc())
        finally:
            progress.close()

    def clear_all_points(self):
        self.image_viewer.points.clear()
        self.points_panel.clean()
        self.image_viewer.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space :
            self.image_viewer.toggle_image_display()
            event.accept()  # 阻止事件继续传递
        if event.key() in [Qt.Key_Up, Qt.Key_Down]:
            curr_row = self.file_view.currentIndex().row()
            if event.key() == Qt.Key_Up:
                curr_row = max(curr_row-1, 0)
            else:
                curr_row = min(self.file_item_model.rowCount()-1, curr_row+1)
            self.file_view.setCurrentIndex(self.file_item_model.index(curr_row, 0))
        else:
            super().keyPressEvent(event)

    def change_result_layer(self, item):
        layer_to_show = [_ for _ in self.layer_to_show]
        if item.isCheckable():
            row = item.row()
            checked = item.checkState() == Qt.Checked
            if checked and row not in layer_to_show:
                layer_to_show.append(row)
                layer_to_show = sorted(layer_to_show)
            elif not checked and row in layer_to_show:
                layer_to_show.pop(layer_to_show.index(row))
            if layer_to_show != self.layer_to_show:
                self.image_viewer.switch_result_layer(layer_to_show)
                self.layer_to_show = layer_to_show

    def set_save_dir(self):
        dialog = QFileDialog(self)
        dialog.setWindowTitle("选择目录")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # 使用Qt对话框

        # 添加自定义控件
        dialog.setOption(QFileDialog.DontResolveSymlinks)
        dialog.setSidebarUrls([QtCore.QUrl.fromLocalFile("/")])  # 添加侧边栏

        if dialog.exec():
            directory = dialog.selectedFiles()
            if directory:
                self.save_dir = directory[0]

    def save_result(self):
        if self.curr_result is None:
            return
        if self.save_dir is None or not os.path.isdir(self.save_dir):
            self.set_save_dir()
        if not os.path.isdir(self.save_dir):
            self.show_status("error", "No save dir")
            return
        result = np.zeros_like(self.curr_result)
        for i in range(self.result_table_model.rowCount()):
            row = self.result_table_model.item(i, 0)
            if row.checkState() == Qt.Checked:
                result[..., 2-i] = self.curr_result[..., i]
        fname = os.path.splitext(os.path.basename(self.curr_file))[0]
        save_path =os.path.join(self.save_dir, f"{fname}.png")
        if cv2.imwrite(save_path, result):
            self.show_status("info", "save to: "+save_path +" success")
        else:
            self.show_status("info", "save to: "+save_path+" failed")


    def show_status(self, level="info", msg= ""):
        info = '{}/{}'.format(
            self.file_view.currentIndex().row()+1,
            self.file_item_model.rowCount()
        )
        # if level == "info":
        #     info += '<div>'
        self.statusBar().showMessage(info)
        if level == "error":
            self.status_label.setText(f'<span style="color: red; font-weight: bold;">{msg}</span>')
        elif level == "warning":
            self.status_label.setText(f'<span style="color: purple; font-weight: bold;">{msg}</span>')
        elif level == "info":
            self.status_label.setText(f'<span style="color: blue; ">{msg}</span>')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)

    # 设置样式
    app.setStyle("Fusion")

    window = SAM2GUI()
    window.show()

    sys.exit(app.exec())
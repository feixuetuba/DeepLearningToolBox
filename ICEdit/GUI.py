import logging
import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from PySide6 import QtCore
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QMouseEvent, QWheelEvent, QStandardItemModel, \
    QStandardItem, QBrush, QIntValidator, QDoubleValidator
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLineEdit, QTextEdit, QPushButton, QCheckBox, QListView, QProgressBar)
from PySide6.QtWidgets import QFileDialog
from diffusers import FluxFillPipeline


class ImageViewer(QWidget):
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

    def set_image(self, cv_img):
        # 读取图像
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w

        self.original_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.display_image = self.original_image
        self._fit_image_to_view()
        self.update()
        self.repaint()

    def set_result_image(self, result):
        # 设置分割结果
        self.result_array = result
        h,w = result.shape[:2]
        bytes_per_line = w << 2
        self.curr_result_array = None

        self.curr_result_array = result
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
                # 右键删除点: 找到最近的点
                if not self.points:
                    return
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


class FocusTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.NoFocus)  # 默认不获取焦点

    def mousePressEvent(self, event):
        #控制QTextEdit何时应当获取Focus
        if event.button() == Qt.LeftButton:
            # 鼠标点击时临时允许获取焦点
            self.setFocusPolicy(Qt.StrongFocus)
            self.setFocus()
        super().mousePressEvent(event)

    def focusOutEvent(self, event):
        # 失去焦点后恢复为不获取焦点
        self.setFocusPolicy(Qt.NoFocus)
        super().focusOutEvent(event)


class ICEditGUI(QMainWindow):
    def __init__(self,
                 flux_path=f"black-forest-labs/FLUX.1-Fill-dev",
                lora_path = "lora/pytorch_lora_weights.safetensors",
                 enable_model_cpu_offload=False,
                 device="cuda:7",
                 title="ICEdit"
                 ):
        super().__init__()
        self.setWindowTitle(title)
        self.title = title
        # 主窗口布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        self.status = ["0/0"]

        self.file_view = QListView(parent=self)
        self.file_item_model = QStandardItemModel()
        self.file_view.setModel(self.file_item_model)
        self.file_view.selectionModel().selectionChanged.connect(self.load_image)
        # self.file_view.setFocusPolicy(Qt.NoFocus)
        self.image_files = []
        main_layout.addWidget(self.file_view, stretch=1)
        int_validator = QIntValidator()
        double_validator = QDoubleValidator()

        # 图像显示区域
        self.image_viewer = ImageViewer()
        self.image_viewer.setFocusPolicy(Qt.NoFocus)
        main_layout.addWidget(self.image_viewer, stretch=3)

        # 右侧控制面板
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(10, 10, 10, 10)

        # 关键点和提示词面板
        self.prompt_editor = FocusTextEdit("prompt")
        # self.prompt_editor.setFocusPolicy(Qt.NoFocus)
        right_panel.addWidget(self.prompt_editor)

        # 操作按钮
        self.load_button = QPushButton("加载图像")
        self.load_button.setFocusPolicy(Qt.NoFocus)
        self.save_button = QPushButton("保存结果")
        self.save_button.setFocusPolicy(Qt.NoFocus)

        self.run_button = QPushButton("运行")
        self.run_button.setFocusPolicy(Qt.NoFocus)
        self.switch_button = QPushButton("显示结果")
        self.switch_button.setFocusPolicy(Qt.NoFocus)
        self.apply_all_button = QPushButton("应用到所有")
        self.apply_all_button.setFocusPolicy(Qt.NoFocus)
        self.no_input = QCheckBox("不保存输入")
        self.no_input.setFocusPolicy(Qt.NoFocus)
        self.seed_edit = QLineEdit("Seed:")
        self.seed_edit.setText("0")
        self.seed_edit.setValidator(int_validator)
        self.step_edit = QLineEdit("Step:")
        self.step_edit.setText("28")
        self.step_edit.setValidator(int_validator)
        self.cfg_edit = QLineEdit("CFG:")
        self.cfg_edit.setText("50")
        self.cfg_edit.setValidator(double_validator)
        self.size_edit = QLineEdit("Size:")
        self.size_edit.setText("512")
        self.size_edit.setValidator(int_validator)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFocusPolicy(Qt.NoFocus)

        right_panel.addWidget(self.load_button)
        right_panel.addWidget(self.save_button)
        right_panel.addWidget(self.run_button)
        right_panel.addWidget(self.switch_button)
        right_panel.addWidget(self.apply_all_button)
        right_panel.addWidget(self.no_input)
        right_panel.addWidget(self.seed_edit)
        right_panel.addWidget(self.step_edit)
        right_panel.addWidget(self.cfg_edit)
        right_panel.addWidget(self.progress_bar)
        right_panel.addStretch()

        main_layout.addLayout(right_panel, stretch=1)

        # 连接信号
        self.load_button.clicked.connect(self.load_files)
        self.save_button.clicked.connect(self.set_result_dir)
        self.run_button.clicked.connect(self.run)
        self.switch_button.clicked.connect(self.toggle_result)
        self.apply_all_button.clicked.connect(self.apply_all)
        self.switch_button.setEnabled(False)

        pipe = FluxFillPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16)
        pipe.load_lora_weights(lora_path)

        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        self.pipe = pipe
        self.curr_file = None
        self.curr_img = None
        self.curr_result = None
        self.latest_save_dir = ""
        self.setGeometry(100, 100, 1200, 800)
        self.dsz = 512

    def load_files(self):
        # 设置文件对话框选项
        dialog = QFileDialog(self)
        dialog.setWindowTitle("选择文件或目录")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # 使用Qt对话框
        dialog.setOption(QFileDialog.ShowDirsOnly, False)
        dialog.setOption(QFileDialog.HideNameFilterDetails, False)
        # 添加自定义控件
        # dialog.setOption(QFileDialog.DontResolveSymlinks)
        dialog.setSidebarUrls([QtCore.QUrl.fromLocalFile("/")])  # 添加侧边栏
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
            self.status[0] = "{}/"+str(self.file_item_model.rowCount())
            self.show_status()

    def show_status(self):
        info = self.status[0].format(self.file_view.currentIndex().row()+1)
        if len(self.status) > 1:
            info = info + " | ".join(self.status[1:])
        self.statusBar().showMessage(info)

    def set_result_dir(self):
        dialog = QFileDialog(self)
        dialog.setWindowTitle("选择保存目录")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # 使用Qt对话框

        # 添加自定义控件
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # 使用Qt对话框
        dialog.setOption(QFileDialog.ShowDirsOnly, False)
        dialog.setOption(QFileDialog.HideNameFilterDetails, False)
        dialog.setOption(QFileDialog.DontResolveSymlinks)
        dialog.setSidebarUrls([QtCore.QUrl.fromLocalFile("/")])  # 添加侧边栏

        dir = dialog.getExistingDirectory()
        if not os.path.isdir(dir):
            return
        self.latest_save_dir = dir
        self.save_result()

    def save_result(self):
        if self.curr_result is None:
            return
        if not os.path.isdir(self.latest_save_dir):
            logging.warning(f"Set directory to save")
            self.set_result_dir()
            return
        file = os.path.basename(self.curr_file)
        fname = os.path.splitext(file)[0]
        if self.no_input.checkState() != Qt.Checked:
            cv2.imwrite(os.path.join(self.latest_save_dir, f"{fname}_in.png"), cv2.cvtColor(self.curr_img, cv2.COLOR_RGBA2BGR))
            cv2.imwrite(os.path.join(self.latest_save_dir, f"{fname}_out.png"), cv2.cvtColor(self.curr_result, cv2.COLOR_RGBA2BGR))
        else:
            cv2.imwrite(f"{self.latest_save_dir}/{fname}_out.png", cv2.cvtColor(self.curr_result, cv2.COLOR_RGBA2BGR))

    def reset(self):
        self.curr_img = self.curr_result = None
        self.curr_file = ""
        self.switch_button.setText("显示结果")
        self.switch_button.setEnabled(False)

    def load_image(self, item):
        self.reset()
        idx = self.file_view.currentIndex().row()
        file_path = self.image_files[idx]["src"]

        # 如果用户选择了文件（没有取消对话框）
        if file_path:
            # 设置图像并重置视图状态
            self.curr_file = file_path
            cv_img = cv2.imread(file_path)
            while max(cv_img.shape) > self.dsz*2:
                cv_img = cv2.resize(cv_img,(0,0),fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            h, w = cv_img.shape[:2]
            s = self.dsz / max(w,h)
            if s < 1.0:
                w = int(w * s)
                h = int(h * s)
                cv_img = cv2.resize(cv_img, (w,h), interpolation=cv2.INTER_AREA)
            self.curr_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            self.image_viewer.current_scale = 1.0  # 重置缩放
            self.image_viewer.image_pos = QPointF(0, 0)  # 重置位置
            self.image_viewer.set_image(self.curr_img)
            # self.clear_all_points()  # 清除之前的点

            fname = os.path.splitext(file_path)[0]
            self.setWindowTitle(f"{self.title} - {fname}")
            self.show_status()

    def run(self):
        self.prompt_editor.setFocusPolicy(Qt.NoFocus)
        # progress = QProgressDialog("正在执行...", "取消", 0, 0, self)
        # progress.setWindowTitle("请稍候")
        # progress.setWindowModality(Qt.WindowModal)
        # progress.setCancelButton(None)  # 移除取消按钮
        # progress.setRange(0, 0)  # 设置为不确定进度模式（环形动画）
        # progress.show()
        # 强制立即显示对话框
        # QApplication.processEvents()
        self.dsz = int(self.size_edit.text())
        if self.dsz <= 0:
            self.dsz = max(self.curr_img.shape[:2]) //8 * 8
        image = Image.fromarray(self.curr_img)
        prompt = self.prompt_editor.toPlainText()
        instruction = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {prompt}'

        width, height = image.size
        combined_image = Image.new("RGB", (2*self.dsz, self.dsz))
        l = (self.dsz - width) >> 1
        t = (self.dsz - height) >> 1
        combined_image.paste(image, (l, t))
        combined_image.paste(image, (self.dsz+l, t))
        mask_array = np.zeros((self.dsz, self.dsz * 2), dtype=np.uint8)
        mask_array[t:t+height, self.dsz+l:self.dsz+l+width] = 255
        mask = Image.fromarray(mask_array)
        try:
            seed = int(self.seed_edit.text())
        except:
            seed = -1
            self.seed_edit.setText("-1")
        if seed < 0:
            seed = random.randint(0, 65535)
        step = int(self.step_edit.text())
        if step <= 0:
            step = 28
        self.progress_bar.setRange(0, step+1)
        self.progress_bar.setValue(0)
        cfg = float(self.cfg_edit.text())
        if cfg <0:
            cfg = 50

        def callback_on_step_end(pipe, step, timestep,
                                 callback_kwarg):
            self.progress_bar.setValue(step+1)
            return callback_kwarg

        result_image = self.pipe(
            prompt=instruction,
            image=combined_image,
            mask_image=mask,
            height=self.dsz,
            width=self.dsz * 2,
            guidance_scale=cfg,
            num_inference_steps=step,
            generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=callback_on_step_end
        ).images[0]
        self.progress_bar.setValue(step+1)
        # progress.close()
        result = np.asarray(result_image)[t:t+height, l+self.dsz:l+self.dsz+width]
        result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
        self.curr_result = result
        self.image_viewer.set_result_image(result)
        self.image_viewer.show_result()
        self.switch_button.setText("显示原图")
        self.switch_button.setEnabled(True)

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

    def toggle_result(self):
        self.image_viewer.toggle_image_display()
        if self.switch_button.text() == "显示原图":
            self.switch_button.setText("显示结果")
        else:
            self.switch_button.setText("显示原图")

    def apply_all(self):
        curr_idx = self.file_view.currentIndex().row()
        total = self.file_item_model.rowCount()
        for i in range(curr_idx, total):
            self.file_view.setCurrentIndex(self.file_item_model.index(i, 0))
            self.run()
            self.save_result()


def select_gpu(min_memory_available=35, verbose=True, use_minium=True):
    """选择可用显存大于阈值的GPU，默认要求至少35GB可用"""
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)  # 临时切换设备以更新统计
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # 已用显存（GB）
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # 总显存（GB）
        free = total - allocated  # 估算可用显存（GB）

        if verbose:
            print(f"GPU {i}: {total:.1f} GB total, {allocated:.1f} GB used, {free:.1f} GB free")

        if free >= min_memory_available:
            available_gpus.append((i, free))
    if len(available_gpus) == 0:
        return -1
    available_gpus = sorted(available_gpus, key=lambda x:x[1])
    if use_minium:
        # 选可用且当前资源最少的，避免浪费
        id = available_gpus[0][0]
    else:
        # 选可用且当前资源最多的，避免崩溃
        id = available_gpus[-1][0]
    return id

if __name__ == "__main__":
    MODEL_DIR = os.environ.get("MODEL_HOME", "")
    gpu_id = select_gpu(35)
    if gpu_id < 0:
        print("至少要35G的显存才能运行， 没有符合条件的显卡")
        exit(-1)
    device = f"cuda:{gpu_id}"

    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)

    # 设置样式
    app.setStyle("Fusion")

    window = ICEditGUI(
        "black-forest-labs/FLUX.1-Fill-dev",
        device=device)
    window.show()

    sys.exit(app.exec())
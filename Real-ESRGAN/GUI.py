import logging
import math
import os
import sys

MDIR = os.path.dirname(os.path.abspath(__file__))
GIT_DIR = os.path.join(MDIR, "Real-ESRGAN")
if not os.path.isdir(GIT_DIR):
    logging.error(f"Real-ESRGAN project dir no found!")
    exit(-1)
sys.path.insert(0, GIT_DIR)
os.chdir(GIT_DIR)

if not os.path.isfile(f"{GIT_DIR}/realesrgan/version.py"):
    os.system("python setup.py develop")

import cv2
import numpy as np
import torch
from PySide6 import QtCore
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QStandardItemModel, \
    QStandardItem, QIntValidator, QDoubleValidator
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLineEdit, QPushButton, QCheckBox, QListView, QProgressBar, QComboBox, QLabel)
from PySide6.QtWidgets import QFileDialog
#pip install basicsr-fixed
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from mutils.QtUi import TitledLineEdit, MImageViewer, getFileSelectDialog


class RealESRGANGUI(QMainWindow):
    def __init__(self,
                 device="cuda:7",
                 title="RealESRGAN"
                 ):
        super().__init__()
        self.setWindowTitle(title)
        self.title = title
        # 主窗口布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

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
        self.image_viewer = MImageViewer()
        self.image_viewer.setFocusPolicy(Qt.NoFocus)
        main_layout.addWidget(self.image_viewer, stretch=3)

        # 右侧控制面板
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(10, 10, 10, 10)

        # 操作按钮
        self.load_button = QPushButton("加载图像")
        self.load_button.setFocusPolicy(Qt.NoFocus)
        self.save_button = QPushButton("保存结果")
        self.save_button.setFocusPolicy(Qt.NoFocus)

        self.model_combox = QComboBox(parent=self)
        self.model_combox.addItems([
            "RealESRGAN_x4plus",
            "RealESRNet_x4plus",
            "RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x2plus",
            "realesr-animevideov3",
            "realesr-general-x4v3"
        ])

        self.run_button = QPushButton("运行")
        self.run_button.setFocusPolicy(Qt.NoFocus)
        self.switch_button = QPushButton("显示结果")
        self.switch_button.setFocusPolicy(Qt.NoFocus)
        self.apply_all_button = QPushButton("应用到所有")
        self.apply_all_button.setFocusPolicy(Qt.NoFocus)
        self.save_input_edit = QCheckBox("保存输入")
        self.save_input_edit.setFocusPolicy(Qt.NoFocus)

        self.slide_size_edit = TitledLineEdit("滑窗尺寸:", self)
        self.slide_size_edit.setText("512")
        self.slide_size_edit.setValidator(int_validator)
        self.slide_overlap_edit = TitledLineEdit("滑窗重叠:", self)
        self.slide_overlap_edit.setText("51")
        self.slide_overlap_edit.setValidator(int_validator)

        self.model_path_edit = TitledLineEdit("模型目录:", self)
        self.model_path_edit.setText("weights")
        self.model_path_edit.setValidator(int_validator)

        self.denoise_strength_edit = TitledLineEdit("去噪强度:", self)
        self.denoise_strength_edit.setText("1.0")
        self.denoise_strength_edit.setValidator(double_validator)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFocusPolicy(Qt.NoFocus)

        right_panel.addWidget(self.load_button)
        right_panel.addWidget(self.save_button)
        right_panel.addWidget(self.run_button)
        right_panel.addWidget(self.switch_button)
        right_panel.addWidget(self.model_combox)
        right_panel.addWidget(self.apply_all_button)
        right_panel.addWidget(self.save_input_edit)
        right_panel.addWidget(self.slide_size_edit)
        right_panel.addWidget(self.slide_overlap_edit)
        right_panel.addWidget(self.denoise_strength_edit)
        # right_panel.addWidget(self.step_edit)
        # right_panel.addWidget(self.cfg_edit)
        right_panel.addWidget(self.model_path_edit)
        right_panel.addWidget(self.progress_bar)
        right_panel.addStretch(1)

        main_layout.addLayout(right_panel, stretch=1)

        colored_label = QLabel()
        colored_label.setTextFormat(Qt.RichText)  # 允许富文本
        self.status_label = colored_label
        # 添加到状态栏的永久区域
        self.statusBar().addPermanentWidget(colored_label)

        # 连接信号
        self.load_button.clicked.connect(self.load_files)
        self.save_button.clicked.connect(self.set_result_dir)
        self.run_button.clicked.connect(self.run)
        self.switch_button.clicked.connect(self.toggle_result)
        self.apply_all_button.clicked.connect(self.apply_all)
        self.switch_button.setEnabled(False)

        self.device = device
        self.model = ("", None)
        self.curr_file = None
        self.curr_img = None
        self.curr_result = None
        self.latest_save_dir = ""
        self.setGeometry(100, 100, 1200, 800)
        self.dsz = 512

    def load_model(self):
        model_name = self.model_combox.currentText()
        if self.model[0] != model_name:
            self.show_status("info", f"{model_name} loading")
            if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
            elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
            elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                netscale = 4
                file_url = [
                    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
            elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                netscale = 2
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
            elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
            elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
                netscale = 4
                file_url = [
                    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
                ]

            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            model_dir = self.model_path_edit.text().strip()
            if model_dir.strip() == "":
                model_dir = "weights"
            if model_dir.startswith(".") or not model_dir.startswith("/"):
                model_dir = os.path.join(ROOT_DIR, model_dir)
            os.makedirs(model_dir, exist_ok=True)
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
            dni_weight = None
            denoise_strength = float(self.denoise_strength_edit.text())
            denoise_strength = max(0, denoise_strength)
            if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
                wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
                model_path = [model_path, wdn_model_path]
                dni_weight = [denoise_strength, 1 - denoise_strength]
            self.model = (model_name, RealESRGANer(
                    scale=netscale,
                    model_path=model_path,
                    dni_weight=dni_weight,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=10,
                    half=False, #not args.fp32,
                    gpu_id=self.device),netscale)#args.gpu_id)
            self.show_status("info", f"{model_name} loaded")

    def load_files(self):
        # 设置文件对话框选项
        dialog = getFileSelectDialog()
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

    def show_status(self, level="info", msg=""):
        info = '{}/{}'.format(
            self.file_view.currentIndex().row() + 1,
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
        if self.save_input_edit.checkState() == Qt.Checked:
            cv2.imwrite(os.path.join(self.latest_save_dir, f"{fname}_in.png"), self.curr_img)
            cv2.imwrite(os.path.join(self.latest_save_dir, f"{fname}_out.png"), self.curr_result)
        else:
            cv2.imwrite(f"{self.latest_save_dir}/{fname}.png", self.curr_result)

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
            self.curr_img = cv_img
            if cv_img.ndim == 1:
                cv_img = cv2.cvtColor(self.curr_img, cv2.COLOR_GRAY2RGB)
            else:
                if cv_img.shape[:2] == 4:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
                else:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            self.image_viewer.current_scale = 1.0  # 重置缩放
            self.image_viewer.image_pos = QPointF(0, 0)  # 重置位置
            self.image_viewer.set_image(cv_img)

            fname = os.path.splitext(file_path)[0]
            self.setWindowTitle(f"{self.title} - {fname}")
            self.show_status()

    def run(self):
        imh, imw = self.curr_img.shape[:2]
        self.load_model()
        upsampler = self.model[1]
        win_sz = max(0, int(self.slide_size_edit.text()))
        win_pad = max(0, int(self.slide_overlap_edit.text()))
        if win_sz == 0:
            win_sz = max(imw, imh)
        if win_pad == 0:
            win_pad = min(imw, imh) // 10
        count_x = math.ceil(imw / win_sz)
        count_y = math.ceil(imh / win_sz)
        scale = self.model[2]
        self.progress_bar.setRange(0, count_x * count_y)
        self.curr_result = np.zeros((imh*scale, imw*scale,3), dtype=np.uint8)
        for y in range(count_y):
            for x in range(count_x):
                # extract tile from input image
                ofs_x = x * win_sz
                ofs_y = y * win_sz
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + win_sz, imw)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + win_sz, imh)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - win_pad, 0)
                input_end_x_pad = min(input_end_x + win_pad, imw)
                input_start_y_pad = max(input_start_y - win_pad, 0)
                input_end_y_pad = min(input_end_y + win_pad, imh)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * count_x + x + 1
                input_tile = self.curr_img[input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile, _ = upsampler.enhance(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                    self.show_status("error", error)
                self.progress_bar.setValue(tile_idx)

                # output tile area on total image
                output_start_x = input_start_x * scale
                output_end_x = input_end_x * scale
                output_start_y = input_start_y * scale
                output_end_y = input_end_y * scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * scale
                output_end_x_tile = output_start_x_tile + input_tile_width * scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * scale
                output_end_y_tile = output_start_y_tile + input_tile_height * scale

                # put tile into output image
                self.curr_result[output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]

        self.image_viewer.set_result_image(cv2.cvtColor(self.curr_result, cv2.COLOR_BGR2RGB))
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
    gpu_id =0
    if gpu_id < 0:
        print("至少要35G的显存才能运行， 没有符合条件的显卡")
        exit(-1)
    device = f"{gpu_id}"

    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)

    # 设置样式
    app.setStyle("Fusion")

    window = RealESRGANGUI(
        device=device)
    window.show()

    sys.exit(app.exec())
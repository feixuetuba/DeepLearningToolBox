import logging
import math
import os
import shutil
import sys
import traceback
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from future.backports import OrderedDict

MDIR = os.path.dirname(os.path.abspath(__file__))
import cv2
import numpy as np
import torch
from PySide6 import QtCore
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QStandardItemModel, \
    QStandardItem, QIntValidator, QDoubleValidator
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLineEdit, QPushButton, QCheckBox, QListView, QProgressBar, QComboBox, QLabel,
                               QTabWidget)
from PySide6.QtWidgets import QFileDialog
import mediapipe as mp

from mutils.QtUi import TitledLineEdit, MImageViewer, getFileSelectDialog, createPannel


def face_align(img, facepoints, expand=[0.3, 0.1, 0.1, 0.1], square=True):
    LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157, 173]  # 左眼
    RIGHT_EYE_INDICES = [362, 398, 384, 385, 386, 387, 388, 466]  # 右眼
    NOSE_TIP_INDEX = 1
    imh, imw = img.shape[:2]
    facepoints = np.array(facepoints) * [imw, imh]
    leye_center = np.mean(facepoints[LEFT_EYE_INDICES],axis=0)
    reye_center = np.mean(facepoints[RIGHT_EYE_INDICES],axis=0)
    dX = reye_center[0] - leye_center[0]
    dY = reye_center[1] - leye_center[1]

    # 计算旋转角度（以弧度为单位）
    angle = np.arctan2(dY, dX)

    # # 将弧度转换为角度
    # angle_deg = np.degrees(angle)

    cos = math.cos(angle)
    sin = math.sin(angle)
    M = np.array([
        [cos, sin, 0],
        [-sin, cos, 0]
    ])
    cos = abs(cos)
    sin = abs(sin)
    neww = imw * cos + imh * sin
    newh = imw * sin + imh * cos
    cx = 0.5 * imw
    cy = 0.5 * imh
    M[0,2] = neww * 0.5 - cx * M[0,0] - cy * M[0,1]
    M[1,2] = newh * 0.5 - cx * M[1,0] - cy * M[1,1]
    invM = cv2.invertAffineTransform(M)
    wlmk = cv2.transform(facepoints[None, ...],M)[0]
    l,t,r,b = wlmk[:,0].min(),wlmk[:,1].min(),wlmk[:,0].max()+1, wlmk[:,1].max()+1
    w = r - l
    h = b - t
    l = max(0, l - w * expand[2])
    r = min(neww, r + w * expand[3])
    t = max(0, t - h * expand[0])
    b = min(newh, b + h * expand[1])
    coors = np.array([
        [l,t], [r,t],
        [l,b], [r,b]
    ])
    inv_coor = cv2.transform(coors[None, ...], invM)[0].astype(int)
    coors = coors.astype(int)
    w = coors[1,0] - coors[0,0]
    h = coors[2,1] - coors[0,1]
    if square:
        w = h = max(w,h)
    l, t, r, b = inv_coor[:, 0].min(), inv_coor[:, 1].min(), inv_coor[:, 0].max(), inv_coor[:, 1].max()
    l = max(0, l)
    t = max(0, t)
    r = min(imw, r)
    b = min(imh, b)
    cx = (r - l) * 0.5
    cy = (b - t) * 0.5
    clmk = facepoints - [l,t]
    M[0,2] = w * 0.5 - cx * M[0,0] - cy * M[0,1]
    M[1,2] = h * 0.5 - cx * M[1,0] - cy * M[1,1]
    face = cv2.warpAffine(img[t:b, l:r], M, (w,h))
    wlmk = cv2.transform(clmk[None, ...], M)
    return face, wlmk, [l, t, r, b]


class MediaPipeGUI(QMainWindow):
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

        self.run_button = QPushButton("运行")
        self.run_button.setFocusPolicy(Qt.NoFocus)
        self.switch_button = QPushButton("显示结果")
        self.switch_button.setFocusPolicy(Qt.NoFocus)
        self.apply_all_button = QPushButton("应用到所有")
        self.apply_all_button.setFocusPolicy(Qt.NoFocus)
        self.save_input_edit = QCheckBox("保存输入")
        self.save_input_edit.setFocusPolicy(Qt.NoFocus)

        self.model_path_edit = TitledLineEdit("模型目录:", self)
        self.model_path_edit.setText("weights")
        self.model_path_edit.setValidator(int_validator)
        self.processors = OrderedDict([
            ("face", [QCheckBox("人脸"),None]),
            ("hand", [QCheckBox("手"),None]),
            ("pose", [QCheckBox("姿态"),None]),
            ("detector", [QCheckBox("检测"),None])
        ])
        self.result_meta = {}
        func_widget = createPannel([_[0] for _ in self.processors.values()], self, "nx3")

        tab_widget = QTabWidget(parent=self)
        self.build_face_tab(tab_widget)
        self.build_hand_tab(tab_widget)
        self.build_pose_tab(tab_widget)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFocusPolicy(Qt.NoFocus)

        right_panel.addWidget(self.load_button)
        right_panel.addWidget(self.save_button)
        right_panel.addWidget(self.run_button)
        right_panel.addWidget(self.switch_button)
        right_panel.addWidget(self.apply_all_button)
        right_panel.addWidget(self.save_input_edit)
        right_panel.addWidget(self.model_path_edit)
        right_panel.addWidget(func_widget)
        right_panel.addWidget(self.progress_bar)
        right_panel.addWidget(tab_widget)
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

    def build_face_tab(self, tab_widget):
        face_w = QWidget()
        layout = QVBoxLayout()
        self.show_face_lmk_cbk = QCheckBox("显示人脸关键点")
        # self.crop_face_cbk = QCheckBox("人脸裁剪")
        self.align_face_cbk = QCheckBox("人脸对齐")

        layout.addWidget(self.show_face_lmk_cbk)
        # layout.addWidget(self.crop_face_cbk)
        layout.addWidget(self.align_face_cbk)
        face_w.setLayout(layout)
        tab_widget.addTab(face_w, "人脸")

    def build_hand_tab(self, tab_widget):
        face_w = QWidget()
        layout = QVBoxLayout()
        self.show_hand_lmk_cbk = QCheckBox("显示手部关键点")

        layout.addWidget(self.show_hand_lmk_cbk)
        face_w.setLayout(layout)
        tab_widget.addTab(face_w, "手")

    def build_pose_tab(self, tab_widget):
        face_w = QWidget()
        layout = QVBoxLayout()
        self.show_pose_lmk_cbk = QCheckBox("显示姿态关键点")

        layout.addWidget(self.show_pose_lmk_cbk)
        face_w.setLayout(layout)
        tab_widget.addTab(face_w, "姿态")

    def load_model(self, model_name, static_image_mode=True):
        #static_image_mode: True处理图像， False处理视频
        if self.processors[model_name][1] is not None:
            return
        if model_name == "face":
            # face_detection = mp.solutions.face_detection.FaceDetection(static_image_mode=static_image_mode, max_num_faces=20, min_detection_confidence=0.5)
            processor = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode, max_num_faces=1, min_detection_confidence=0.5)
            # face_mesh_connections = mp.solutions.face_mesh_connections(static_image_mode=static_image_mode, max_num_faces=1, min_detection_confidence=0.5)
        elif model_name == "hand":
            hand_detection = mp.solutions.hands.Hands(static_image_mode=static_image_mode,
                                                      max_num_hands=20,
                                                      min_detection_confidence=0.5 )
            processor = (hand_detection)
        elif model_name == "objectron":
            obj = mp.solutions.objectron.Objectron(static_image_mode,)
            processor = (obj)
        elif model_name == "pose":
            BaseOptions = python.BaseOptions
            PoseLandmarker = vision.PoseLandmarker
            PoseLandmarkerOptions = vision.PoseLandmarkerOptions
            RunningMode = vision.RunningMode

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=os.path.join(MDIR, "weights", "pose_landmarker_full.task")),
                running_mode=RunningMode.IMAGE,
                num_poses=10,
                min_pose_detection_confidence=0.6

            )
            landmarker = PoseLandmarker.create_from_options(options)
            # pose = mp.tasks.pose.Pose(static_image_mode=static_image_mode, model_complexity=1,min_detection_confidence=0.5,  # 检测置信度阈值
            #                                             min_tracking_confidence=0.5,   # 跟踪置信度阈值
            #                                             num_poses=5 )
            processor = (landmarker)
        else:
            raise  ValueError(f"Unkonw {model_name}")
        self.processors[model_name][1] = processor

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

            self.image_viewer.reset()
            self.image_viewer.set_image(cv_img)

            fname = os.path.splitext(file_path)[0]
            self.setWindowTitle(f"{self.title} - {fname}")
            self.show_status()

    def call_face_processor(self, img):
        results = self.processors["face"][1].process(img)
        ret = []
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                curr = []
                for landmark in landmarks.landmark:
                    curr.append([landmark.x, landmark.y])
                ret.append(curr)
        return ret

    def call_hand_processor(self, img):
        results = self.processors["hand"][1].process(img)
        ret = []
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                curr = []
                for landmark in landmarks.landmark:
                    # 将归一化坐标转换为像素坐标
                    # h, w, _ = frame.shape
                    # x, y = int(landmark.x * w), int(landmark.y * h)
                    curr.append([landmark.x, landmark.y])
                ret.append(curr)
        return ret

    def call_pose_processor(self, img):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        results = self.processors["pose"][1].detect(mp_image)
        ret = []
        if results.pose_landmarks:
            for pose_landmarks  in results.pose_landmarks:
                curr = []
                for landmark in pose_landmarks :
                    # 将归一化坐标转换为像素坐标
                    # h, w, _ = frame.shape
                    # x, y = int(landmark.x * w), int(landmark.y * h)
                    curr.append([landmark.x, landmark.y])
                ret.append(curr)
            self.show_status("info", f"{len(results.pose_landmarks)} Pose detected")
        return ret

    def show_result(self):
        result = self.curr_img.copy()
        imh, imw = result.shape[:2]
        if "face" in self.result_meta:
            if self.show_face_lmk_cbk.checkState() == Qt.Checked:
                for kpts in self.result_meta["face"]:
                    for i, (x, y) in enumerate(kpts):
                        x = int(imw * x)
                        y = int(imh * y)
                        result = cv2.circle(result, (x, y), (3), (0, 0, 255, 255), -1)
                        # result = cv2.putText(result, str(i), (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255,0,0), 1)
            if self.align_face_cbk.checkState() == Qt.Checked:
                faces = []
                maxh, maxw = 0,0
                for kpts in self.result_meta["face"]:
                    face, wlmk, face_roi =  face_align(result,kpts)
                    faces.append(face)
                    maxh = max(maxh, face.shape[0])
                    maxw = max(maxw, face.shape[1])

                if len(faces) > 1:
                    for i, face in enumerate(faces):
                        h, w = face.shape[:2]
                        pads =[maxh - h, 0, maxw - w, 0]
                        pads[1] = pads[0] >> 1
                        pads[3] = pads[2] >> 1
                        pads[0] -= pads[1]
                        pads[2] -= pads[3]
                        face = cv2.copyMakeBorder(face, *pads, cv2.BORDER_CONSTANT, value=0)
                        faces[i] = face
                    col = math.ceil(len(faces)**0.5)
                    row = len(faces) / col
                    result = np.zeros((row * maxh, col * maxw,3), dtype=np.uint8)
                    for y in range(row):
                        y = y * maxh
                        for x in range(col):
                            x = x * maxw
                            result[y:y+maxh, x:x+maxw] =  faces[i]
                else:
                    result = faces[0]

        if "hand" in self.result_meta:
            if self.show_hand_lmk_cbk.checkState() == Qt.Checked:
                for kpts in self.result_meta["hand"]:
                    for i, (x, y) in enumerate(kpts):
                        x = int(imw * x)
                        y = int(imh * y)
                        result = cv2.circle(result, (x, y), (3), (0, 0, 255, 255), -1)
                        result = cv2.putText(result, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1)
        if "pose" in self.result_meta:
            if self.show_pose_lmk_cbk.checkState() == Qt.Checked:
                for kpts in self.result_meta["pose"]:
                    for i, (x, y) in enumerate(kpts):
                        x = int(imw * x)
                        y = int(imh * y)
                        result = cv2.circle(result, (x, y), (3), (0, 0, 255, 255), -1)
                        result = cv2.putText(result, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1)
        self.curr_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        self.image_viewer.set_result_image(self.curr_result)
        # self.image_viewer.show_result()
        # self.switch_button.setText("显示原图")
        self.switch_button.setEnabled(True)

    def run(self):
        imh, imw = self.curr_img.shape[:2]
        netin = cv2.cvtColor(self.curr_img[...,:3], cv2.COLOR_BGR2RGB)
        result = self.curr_img.copy()
        self.result_meta = {}
        for name, value in self.processors.items():
            if value[0].checkState() != Qt.Checked:
                continue
            self.load_model(name, True)
            self.result_meta[name] = getattr(self, f"call_{name}_processor")(netin)
        self.curr_result = self.curr_img.copy()
        self.show_result()


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
        if self.switch_button.text() == "显示原图":
            self.image_viewer.toggle_image_display()
            self.switch_button.setText("显示结果")
        else:
            self.show_result()
            self.image_viewer.toggle_image_display()
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
    device = f"{gpu_id}"

    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)

    # 设置样式
    app.setStyle("Fusion")

    window = MediaPipeGUI(
        device=device)
    window.show()

    sys.exit(app.exec())
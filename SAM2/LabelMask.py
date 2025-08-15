#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySide6 + SAM2 辅助人体部位标注工具（单文件加强版）
新增/修复要点：
1) 支持删除“单个 Prompt 点 / 单个 Box”：
   - Prompt 模式下：Alt + 左键 删除最近点；右键菜单可删最近点
   - Box 模式下：Alt + 左键 删除包含点的 Box（否则删最近的）；右键菜单可删包含/最近框
2) Redo/Undo 修复：采用 begin/end 快照式命令，Brush/Lasso/SmartSelect/ApplyCandidate 均可撤销重做
3) SAM2 需要同时配置 model_cfg（配置文件）与 model_ckpt（权重）；配置与 App 状态统一保存到 JSON
   - 配置路径：~/.sam2_labeler_config.json，启动时加载，修改后即时写回

其它能力（保留自上版）：
- 多文件懒加载 + 左侧列表
- 自动保存（可开关）/ 每图保存到同名目录（labels.png + 每类二值图 + labels.json）
- 标签与颜色可配置
- 工具：Prompt/Box/Brush/Eraser/Lasso/SmartSelect
- 候选掩码（SAM）→ Apply to Class；不同工具不同鼠标指针
- 批量“应用到所有”
"""

from __future__ import annotations
import os, sys, json, pathlib
MDIR = os.path.dirname(os.path.abspath(__file__))
SAM2= os.path.join(MDIR, "sam2")
sys.path.insert(0,SAM2)
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from PIL import ImageDraw

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QAction, QColor, QImage, QPixmap, QPainter, QPen, QBrush,
    QPainterPath, QCursor, QKeySequence, QUndoStack, QUndoCommand
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsRectItem, QComboBox, QSpinBox,
    QSlider, QFrame, QSplitter, QColorDialog, QDialog, QDialogButtonBox,
    QInputDialog, QMenu, QGraphicsPathItem
)

# ----------------------------- 常量与默认 -----------------------------
APP_NAME = "SAM2_HumanPart_Labeler"
CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".sam2_labeler_config.json")

DEFAULT_CLASSES = [
    {"id": 1, "name": "Head",  "color": [255, 64, 64, 160]},
    {"id": 2, "name": "Torso", "color": [64, 192, 255, 160]},
    {"id": 3, "name": "Arms",  "color": [64, 255, 128, 160]},
    {"id": 4, "name": "Legs",  "color": [255, 192, 64, 160]},
]
DEFAULT_ALPHA = 160
TOOLS = ("prompt", "box", "brush", "eraser", "lasso", "smart")

# ----------------------------- SAM/SAM2 适配器 -----------------------------
class PredictorAdapter:
    """统一封装 SAM 与 SAM2 的最小推理接口。"""
    def __init__(self, model_type: str = "sam2", checkpoint_path: Optional[str] = None, model_cfg: Optional[str] = None):
        self.model_type = (model_type or "sam2").lower()
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.backend = None
        self.image_set = False
        self._try_init_backend()

    def _try_init_backend(self):
        try:
            if self.model_type == "sam2":
                # SAM2 需要 cfg + ckpt
                if not (self.model_cfg and os.path.isfile(self.model_cfg) and self.checkpoint_path and os.path.isfile(self.checkpoint_path)):
                    raise RuntimeError("SAM2 需要配置文件(model_cfg)与权重(model_ckpt)")
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                sam = build_sam2(self.model_cfg, self.checkpoint_path)
                self.backend = SAM2ImagePredictor(sam)
            else:
                # SAM v1 只需要 ckpt
                from segment_anything import sam_model_registry, SamPredictor
                if not (self.checkpoint_path and os.path.isfile(self.checkpoint_path)):
                    raise RuntimeError("SAM(v1) 需要提供 checkpoint_path")
                # 这里默认 vit_h，可按需改成 vit_b / vit_l
                sam = sam_model_registry["vit_h"](checkpoint=self.checkpoint_path)
                self.backend = SamPredictor(sam)
        except Exception as e:
            print("[PredictorAdapter] backend init failed:", e)
            self.backend = None

    def set_image(self, image_rgb: np.ndarray):
        if self.backend is None:
            self.image_set = False
            return
        try:
            self.backend.set_image(image_rgb)
            self.image_set = True
        except Exception as e:
            print("[PredictorAdapter] set_image failed:", e)
            self.image_set = False

    def predict(self,
                points_xy: Optional[np.ndarray],
                points_label: Optional[np.ndarray],
                box_xyxy: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if self.backend is None or not self.image_set:
            return None
        try:
            masks, scores, _ = self.backend.predict(
                point_coords=points_xy if points_xy is not None else None,
                point_labels=points_label if points_label is not None else None,
                box=box_xyxy if box_xyxy is not None else None,
                multimask_output=True,
            )
            if masks is None or len(masks) == 0:
                return None
            best = int(np.argmax(scores))
            m = masks[best]
            if m.ndim == 3:
                m = m[0]
            return (m > 0.5).astype(np.bool_)
        except Exception as e:
            print("[PredictorAdapter] predict failed:", e)
            return None

# ----------------------------- 简易智能选取（颜色连通） -----------------------------
def smart_select(image: np.ndarray, seed_xy: Tuple[int,int], tol: int = 25) -> np.ndarray:
    h, w = image.shape[:2]
    x, y = max(0, min(w-1, seed_xy[0])), max(0, min(h-1, seed_xy[1]))
    seed = image[y, x].astype(np.int32)
    diff = image.astype(np.int32) - seed
    dist = np.sqrt(np.sum(diff*diff, axis=2))
    thresh = dist <= tol
    if not thresh[y, x]:
        return np.zeros((h,w), dtype=bool)
    visited = np.zeros_like(thresh, dtype=bool)
    out = np.zeros_like(thresh, dtype=bool)
    stack = [(y, x)]
    visited[y, x] = True; out[y, x] = True
    while stack:
        cy, cx = stack.pop()
        for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and thresh[ny, nx]:
                visited[ny, nx] = True; out[ny, nx] = True; stack.append((ny, nx))
    return out

# ----------------------------- 图像/绘制工具 -----------------------------
def qimage_from_rgb(rgb: np.ndarray) -> QImage:
    h, w = rgb.shape[:2]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888).copy()

def colorize_mask(mask: np.ndarray, rgba: Tuple[int,int,int,int]) -> QImage:
    h, w = mask.shape
    R, G, B, A = rgba
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[mask, 0] = R; out[mask, 1] = G; out[mask, 2] = B; out[mask, 3] = A
    return QImage(out.data, w, h, 4*w, QImage.Format_RGBA8888).copy()

def paint_disk(mask: np.ndarray, center_rc: Tuple[int, int], radius: int, value: bool):
    r0, c0 = center_rc
    h, w = mask.shape
    r = max(1, int(radius))
    rr = np.arange(max(0, r0 - r), min(h, r0 + r + 1))
    cc = np.arange(max(0, c0 - r), min(w, c0 + r + 1))
    yy, xx = np.meshgrid(rr, cc, indexing='ij')
    disk = (yy - r0)**2 + (xx - c0)**2 <= r*r
    sub = mask[rr.min():rr.max()+1, cc.min():cc.max()+1]
    sub[disk] = value
    mask[rr.min():rr.max()+1, cc.min():cc.max()+1] = sub

def rasterize_polygon_mask(points: List[Tuple[float,float]], hw: Tuple[int,int]) -> np.ndarray:
    """
        使用 PIL.ImageDraw.polygon 精确光栅化多边形到 mask。
        points: list of (x, y) in scene/image coordinates (float OK)
        hw: (h, w)
        返回布尔数组 shape (h, w) 与 image 像素一一对应
        """
    h, w = hw
    if len(points) < 3:
        return np.zeros((h, w), dtype=bool)
    # PIL expects (width, height)
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    xy = [(float(x), float(y)) for (x, y) in points]
    # fill polygon
    draw.polygon(xy, outline=1, fill=1)
    arr = np.array(img, dtype=np.uint8)
    return arr > 0
    # h, w = hw
    # if len(points) < 3:
    #     return np.zeros((h,w), dtype=bool)
    # qimg = QImage(w, h, QImage.Format_Grayscale8)
    # qimg.fill(0)
    # painter = QPainter(qimg)
    # painter.setPen(Qt.NoPen)
    # painter.setBrush(Qt.white)
    # path = QPainterPath()
    # path.moveTo(points[0][0], points[0][1])
    # for (x, y) in points[1:]:
    #     path.lineTo(x, y)
    # path.closeSubpath()
    # painter.drawPath(path)
    # painter.end()
    # arr = np.frombuffer(qimg.constBits(), dtype=np.uint8, count=w*h).reshape((h, w))
    # return arr > 0

# ----------------------------- 数据结构 -----------------------------
@dataclass
class ClickPoint:
    pos: Tuple[float, float]
    label: int  # 1 pos / 0 neg
    item: Optional[QGraphicsEllipseItem] = None

@dataclass
class BoxEntry:
    rect: Tuple[float,float,float,float]  # x0,y0,x1,y1
    item: QGraphicsRectItem

@dataclass
class ClassDef:
    id: int
    name: str
    color: Tuple[int,int,int,int]

@dataclass
class ImageDoc:
    path: str
    size_hw: Tuple[int,int] = (0,0)
    image_rgb: Optional[np.ndarray] = None
    base_pixmap: Optional[QPixmap] = None
    masks: Dict[int, np.ndarray] = field(default_factory=dict)
    dirty: bool = False

# ----------------------------- 交互视图 -----------------------------
class CanvasView(QGraphicsView):
    """负责鼠标交互；通过回调把事件抛给 MainWindow。"""
    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setMouseTracking(True)
        self.mode = "prompt"
        self.brush_radius = 16
        self.smart_tol = 25

        # 回调
        self.cb_prompt_click = None         # (pos_xy,label)
        self.cb_box_update = None           # (x0,y0,x1,y1)
        self.cb_box_finish = None           # (x0,y0,x1,y1)
        self.cb_brush_draw = None           # (pos_xy,radius,add)
        self.cb_stroke_begin = None
        self.cb_stroke_end = None
        self.cb_smart_click = None          # (pos_xy,tol)
        self.cb_lasso_update = None         # (points)
        self.cb_lasso_finish = None         # (points)
        self.cb_delete_point_at = None      # (pos_xy)
        self.cb_delete_box_at = None        # (pos_xy)
        self.cb_clear_prompts = None

        # 内部状态
        self._dragging_box = False
        self._box_start = None
        self._stroking = False
        self._lasso_points: List[Tuple[float,float]] = []

        self._set_cursor()

        # 自定义右键菜单
        self.setContextMenuPolicy(Qt.DefaultContextMenu)

    def set_mode(self, mode: str):
        self.mode = mode
        if mode in ("brush", "eraser", "lasso", "smart", "prompt"):
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._set_cursor()

    def _set_cursor(self):
        if self.mode == "prompt":
            self.viewport().setCursor(QCursor(Qt.PointingHandCursor))
        elif self.mode == "box":
            self.viewport().setCursor(QCursor(Qt.CrossCursor))
        elif self.mode == "brush":
            self.viewport().setCursor(QCursor(Qt.CrossCursor))
        elif self.mode == "eraser":
            self.viewport().setCursor(QCursor(Qt.ForbiddenCursor))
        elif self.mode == "lasso":
            self.viewport().setCursor(QCursor(Qt.CrossCursor))
        elif self.mode == "smart":
            self.viewport().setCursor(QCursor(Qt.WhatsThisCursor))
        else:
            self.viewport().unsetCursor()

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            factor = factor * factor
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        pos_scene = self.mapToScene(event.position().toPoint())
        x, y = pos_scene.x(), pos_scene.y()

        # Alt 快捷删除
        if event.modifiers() & Qt.AltModifier:
            if event.button() == Qt.LeftButton:
                if self.mode == "prompt" and self.cb_delete_point_at:
                    self.cb_delete_point_at((x, y))
                    return
                if self.mode == "box" and self.cb_delete_box_at:
                    self.cb_delete_box_at((x, y))
                    return

        if event.button() == Qt.LeftButton:
            if self.mode == "prompt":
                if self.cb_prompt_click:
                    self.cb_prompt_click((x, y), 1)
            elif self.mode == "box":
                self._dragging_box = True
                self._box_start = (x, y)
            elif self.mode == "brush":
                self._stroking = True
                if self.cb_stroke_begin: self.cb_stroke_begin()
                if self.cb_brush_draw: self.cb_brush_draw((x, y), self.brush_radius, True)
            elif self.mode == "eraser":
                self._stroking = True
                if self.cb_stroke_begin: self.cb_stroke_begin()
                if self.cb_brush_draw: self.cb_brush_draw((x, y), self.brush_radius, False)
            elif self.mode == "smart":
                if self.cb_smart_click: self.cb_smart_click((x, y), self.smart_tol)
            elif self.mode == "lasso":
                self._lasso_points = [(x, y)]
                if self.cb_lasso_update: self.cb_lasso_update(self._lasso_points)
        elif event.button() == Qt.RightButton:
            pass
            # if self.mode == "prompt":
            #     if self.cb_prompt_click:
            #         self.cb_prompt_click((x, y), 0)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos_scene = self.mapToScene(event.position().toPoint())
        x, y = pos_scene.x(), pos_scene.y()
        if self.mode == "box" and self._dragging_box and self._box_start is not None:
            x0, y0 = self._box_start
            if self.cb_box_update: self.cb_box_update((x0, y0, x, y))
        elif self.mode == "brush":
            if self._stroking and (event.buttons() & Qt.LeftButton):
                if self.cb_brush_draw: self.cb_brush_draw((x, y), self.brush_radius, True)
        elif self.mode == "eraser":
            if self._stroking and (event.buttons() & (Qt.LeftButton | Qt.RightButton)):
                if self.cb_brush_draw: self.cb_brush_draw((x, y), self.brush_radius, False)
        elif self.mode == "lasso":
            if event.buttons() & Qt.LeftButton and len(self._lasso_points) > 0:
                self._lasso_points.append((x, y))
                if self.cb_lasso_update: self.cb_lasso_update(self._lasso_points)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        pos_scene = self.mapToScene(event.position().toPoint())
        x, y = pos_scene.x(), pos_scene.y()
        if event.button() == Qt.LeftButton:
            if self.mode == "box" and self._dragging_box and self._box_start is not None:
                x0, y0 = self._box_start
                if self.cb_box_finish: self.cb_box_finish((x0, y0, x, y))
                self._dragging_box = False
                self._box_start = None
            elif self.mode in ("brush", "eraser"):
                if self._stroking and self.cb_stroke_end:
                    self.cb_stroke_end()
                self._stroking = False
            elif self.mode == "lasso" and len(self._lasso_points) >= 3:
                if self.cb_lasso_finish:
                    self.cb_lasso_finish(self._lasso_points)
                self._lasso_points = []
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        pos_scene = self.mapToScene(event.pos())
        menu = QMenu(self)
        act_del_pt = menu.addAction("删除最近点")
        act_del_box = menu.addAction("删除包含/最近框")
        menu.addSeparator()
        act_clear = menu.addAction("清除所有 Prompts")
        chosen = menu.exec(self.mapToGlobal(event.pos()))
        if chosen == act_del_pt and self.cb_delete_point_at:
            self.cb_delete_point_at((pos_scene.x(), pos_scene.y()))
        elif chosen == act_del_box and self.cb_delete_box_at:
            self.cb_delete_box_at((pos_scene.x(), pos_scene.y()))
        elif chosen == act_clear and self.cb_clear_prompts:
            self.cb_clear_prompts()

# ----------------------------- 主窗口 -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 + SAM2 人体部位标注工具")
        self.resize(1440, 900)

        # 状态
        self.cfg = self._load_or_init_config()
        self.classes: Dict[int, ClassDef] = {}
        self.class_order: List[int] = []
        for d in self.cfg.get("classes", DEFAULT_CLASSES):
            cid = int(d["id"]); name = d["name"]; color = tuple(int(x) for x in d["color"])
            self.classes[cid] = ClassDef(cid, name, color); self.class_order.append(cid)

        self.predictor = PredictorAdapter(
            model_type=self.cfg.get("model_type", "sam2"),
            checkpoint_path=self.cfg.get("model_ckpt") or None,
            model_cfg=self.cfg.get("model_cfg") or None
        )

        self.docs: List[ImageDoc] = []
        self.cur_index: int = -1

        # Prompts
        self.clicks: List[ClickPoint] = []
        self.boxes: List[BoxEntry] = []         # 支持多框
        self.box_preview: Optional[QGraphicsRectItem] = None
        self.candidate_mask: Optional[np.ndarray] = None
        self.lasso_path_item: Optional[QGraphicsPathItem] = None

        # 撤销/重做
        self.undo_stack = QUndoStack(self)
        self._capture_before: Optional[Dict[int, np.ndarray]] = None  # 快照起点

        # 场景与图层
        self.scene = QGraphicsScene(self)
        self.view = CanvasView(self.scene, self)

        self.base_item = QGraphicsPixmapItem()
        self.overlay_item = QGraphicsPixmapItem()
        self.candidate_item = QGraphicsPixmapItem()

        self.scene.addItem(self.base_item)
        self.scene.addItem(self.overlay_item)
        self.scene.addItem(self.candidate_item)

        # 左侧文件列表
        self.list_files = QListWidget()
        self.list_files.currentRowChanged.connect(self._on_select_index)

        # 右侧控制面板
        self.right_panel = self._build_right_panel()

        # 布局
        splitter = QSplitter()
        left_panel = QWidget(); lv = QVBoxLayout(left_panel); lv.setContentsMargins(6,6,6,6)
        btn_add_files = QPushButton("添加文件"); btn_add_files.clicked.connect(self._add_files)
        btn_add_dir = QPushButton("添加文件夹"); btn_add_dir.clicked.connect(self._add_dir)
        lv.addWidget(btn_add_files); lv.addWidget(btn_add_dir)
        lv.addWidget(self.list_files, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(self.view)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)

        # 菜单
        self._build_menu()

        # 绑定视图回调
        self.view.cb_prompt_click = self._on_prompt_click
        self.view.cb_box_update = self._on_box_update
        self.view.cb_box_finish = self._on_box_finish
        self.view.cb_brush_draw = self._on_brush_draw
        self.view.cb_stroke_begin = self._begin_mask_edit
        self.view.cb_stroke_end = lambda: self._end_mask_edit("Brush/Eraser")
        self.view.cb_smart_click = self._on_smart_click
        self.view.cb_lasso_update = self._on_lasso_update
        self.view.cb_lasso_finish = self._on_lasso_finish
        self.view.cb_delete_point_at = self._delete_nearest_point
        self.view.cb_delete_box_at = self._delete_box_at
        self.view.cb_clear_prompts = self._clear_prompts

        self._refresh_ui_state()

    # ---------------------- 配置持久化 ----------------------
    def _default_config(self):
        return {
            "model_type": "sam2",
            "model_cfg": "",    # 新增：SAM2 配置文件
            "model_ckpt": "",
            "save_dir": "",
            "auto_save": True,
            "classes": DEFAULT_CLASSES,
        }

    def _load_or_init_config(self):
        if os.path.isfile(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print("加载配置失败，将使用默认：", e)
        # 无文件或失败→默认，并写回
        cfg = self._default_config()
        self._write_config(cfg)
        return cfg

    def _write_config(self, cfg=None):
        data = cfg if cfg is not None else {
            "model_type": self.cfg.get("model_type", "sam2"),
            "model_cfg": self.cfg.get("model_cfg", ""),
            "model_ckpt": self.cfg.get("model_ckpt", ""),
            "save_dir": self.cfg.get("save_dir", ""),
            "auto_save": bool(self.cfg.get("auto_save", True)),
            "classes": [{"id": self.classes[cid].id, "name": self.classes[cid].name, "color": list(self.classes[cid].color)} for cid in self.class_order],
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---------------------- 菜单 ----------------------
    def _build_menu(self):
        m_file = self.menuBar().addMenu("文件")
        act_export = QAction("导出当前图片", self); act_export.triggered.connect(self._export_current)
        act_quit = QAction("退出", self); act_quit.triggered.connect(self.close)
        m_file.addAction(act_export); m_file.addSeparator(); m_file.addAction(act_quit)

        m_set = self.menuBar().addMenu("设置")
        act_model = QAction("设置模型...", self); act_model.triggered.connect(self._set_model)
        act_save_dir = QAction("设置保存路径...", self); act_save_dir.triggered.connect(self._set_save_dir)
        act_auto = QAction("切换自动保存", self, checkable=True); act_auto.setChecked(bool(self.cfg.get("auto_save", True)))
        act_auto.toggled.connect(lambda v: self._toggle_auto_save(v))
        act_labels = QAction("标签与颜色...", self); act_labels.triggered.connect(self._manage_labels)
        m_set.addAction(act_model); m_set.addAction(act_save_dir); m_set.addAction(act_auto); m_set.addAction(act_labels)

        m_edit = self.menuBar().addMenu("编辑")
        act_undo = QAction("撤销", self); act_undo.setShortcut(QKeySequence.Undo); act_undo.triggered.connect(self.undo_stack.undo)
        act_redo = QAction("重做", self); act_redo.setShortcut(QKeySequence.Redo); act_redo.triggered.connect(self.undo_stack.redo)
        act_clear_prompts = QAction("清除 Prompts", self); act_clear_prompts.triggered.connect(self._clear_prompts)
        m_edit.addAction(act_undo); m_edit.addAction(act_redo); m_edit.addSeparator(); m_edit.addAction(act_clear_prompts)

        m_batch = self.menuBar().addMenu("批处理")
        act_apply_all = QAction("将当前SAM提示应用到所有图像", self); act_apply_all.triggered.connect(self._apply_sam_to_all)
        m_batch.addAction(act_apply_all)

    # ---------------------- 右侧面板 ----------------------
    def _build_right_panel(self) -> QWidget:
        w = QWidget(); layout = QVBoxLayout(w); layout.setContentsMargins(8,8,8,8); layout.setSpacing(8)

        layout.addWidget(self._hline("工具"))
        row = QHBoxLayout()
        self.cmb_tool = QComboBox(); self.cmb_tool.addItems(["Prompt", "Box", "Brush", "Eraser", "Lasso", "SmartSelect"])
        self.cmb_tool.currentIndexChanged.connect(self._on_tool_changed)
        row.addWidget(QLabel("Tool:")); row.addWidget(self.cmb_tool, 1)
        layout.addLayout(row)

        hb = QHBoxLayout()
        self.spin_radius = QSpinBox(); self.spin_radius.setRange(1, 256); self.spin_radius.setValue(16)
        self.spin_radius.valueChanged.connect(lambda v: setattr(self.view, "brush_radius", int(v)))
        hb.addWidget(QLabel("半径")); hb.addWidget(self.spin_radius)
        self.slider_tol = QSlider(Qt.Horizontal); self.slider_tol.setRange(1, 80); self.slider_tol.setValue(25)
        self.slider_tol.valueChanged.connect(lambda v: setattr(self.view, "smart_tol", int(v)))
        hb.addWidget(QLabel("智能选取阈值")); hb.addWidget(self.slider_tol, 1)
        layout.addLayout(hb)

        layout.addWidget(self._hline("标签与颜色"))
        self.cmb_class = QComboBox(); self._refresh_class_combo()
        layout.addWidget(self.cmb_class)
        btn_color = QPushButton("设置当前类别颜色"); btn_color.clicked.connect(self._set_current_class_color)
        layout.addWidget(btn_color)

        layout.addWidget(self._hline("SAM 辅助"))
        row2 = QHBoxLayout()
        self.lbl_pts = QLabel("Points: 0(+)/0(-)")
        self.lbl_boxes = QLabel("Boxes: 0")
        btn_clear = QPushButton("Clear Prompts"); btn_clear.clicked.connect(self._clear_prompts)
        row2.addWidget(self.lbl_pts); row2.addWidget(self.lbl_boxes, 1); row2.addWidget(btn_clear)
        layout.addLayout(row2)
        row3 = QHBoxLayout()
        self.btn_run_sam = QPushButton("Run SAM"); self.btn_run_sam.clicked.connect(self._run_sam)
        self.btn_apply = QPushButton("Apply to Class"); self.btn_apply.clicked.connect(self._apply_candidate_to_class)
        row3.addWidget(self.btn_run_sam); row3.addWidget(self.btn_apply)
        layout.addLayout(row3)

        layout.addWidget(self._hline("叠加透明度"))
        self.slider_alpha = QSlider(Qt.Horizontal); self.slider_alpha.setRange(0, 255); self.slider_alpha.setValue(DEFAULT_ALPHA)
        self.slider_alpha.valueChanged.connect(lambda _: self._refresh_overlay())
        layout.addWidget(self.slider_alpha)

        layout.addStretch(1)
        return w

    def _hline(self, text: str) -> QWidget:
        line = QFrame(); line.setFrameShape(QFrame.HLine)
        wrap = QWidget(); l = QHBoxLayout(wrap); l.setContentsMargins(0,0,0,0)
        lab = QLabel(text); lab.setStyleSheet("font-weight:600")
        l.addWidget(lab); l.addWidget(line, 1)
        return wrap

    # ---------------------- 类别管理 ----------------------
    def _refresh_class_combo(self):
        self.cmb_class.blockSignals(True); self.cmb_class.clear()
        for cid in self.class_order:
            self.cmb_class.addItem(f"{cid}: {self.classes[cid].name}", userData=cid)
        self.cmb_class.blockSignals(False)

    def _manage_labels(self):
        dlg = QDialog(self); dlg.setWindowTitle("标签与颜色管理")
        lay = QVBoxLayout(dlg)
        lst = QListWidget()
        for cid in self.class_order:
            it = QListWidgetItem(f"{cid}: {self.classes[cid].name}"); it.setData(Qt.UserRole, cid); lst.addItem(it)
        lay.addWidget(lst)

        row = QHBoxLayout()
        b_add = QPushButton("添加"); b_del = QPushButton("删除"); b_ren = QPushButton("重命名"); b_col = QPushButton("颜色")
        row.addWidget(b_add); row.addWidget(b_del); row.addWidget(b_ren); row.addWidget(b_col)
        lay.addLayout(row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
        lay.addWidget(bb)

        def add_label():
            name, ok = QInputDialog.getText(dlg, "新建标签", "名称：")
            if not ok or not name.strip():
                return
            new_id = (max(self.class_order) + 1) if self.class_order else 1
            self.classes[new_id] = ClassDef(new_id, name.strip(), (255,255,255,160))
            self.class_order.append(new_id)
            it = QListWidgetItem(f"{new_id}: {name.strip()}"); it.setData(Qt.UserRole, new_id); lst.addItem(it)

        def del_label():
            it = lst.currentItem();
            if not it:
                return
            cid = int(it.data(Qt.UserRole))
            if QMessageBox.question(dlg, "确认", f"删除标签 {self.classes[cid].name}?") != QMessageBox.Yes:
                return
            self.class_order = [x for x in self.class_order if x != cid]
            self.classes.pop(cid, None)
            lst.takeItem(lst.row(it))

        def rename_label():
            it = lst.currentItem()
            if not it:
                return
            cid = int(it.data(Qt.UserRole))
            name, ok = QInputDialog.getText(dlg, "重命名", "名称：", text=self.classes[cid].name)
            if not ok or not name.strip():
                return
            self.classes[cid].name = name.strip()
            it.setText(f"{cid}: {name.strip()}")

        def recolor_label():
            it = lst.currentItem();
            if not it:
                return
            cid = int(it.data(Qt.UserRole))
            c = self.classes[cid].color
            qc = QColorDialog.getColor(QColor(c[0], c[1], c[2]), dlg, "选择颜色")
            if not qc.isValid():
                return
            self.classes[cid].color = (qc.red(), qc.green(), qc.blue(), c[3])

        b_add.clicked.connect(add_label); b_del.clicked.connect(del_label)
        b_ren.clicked.connect(rename_label); b_col.clicked.connect(recolor_label)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)

        if dlg.exec():
            self._refresh_class_combo()
            self.cfg["classes"] = [{"id": self.classes[cid].id, "name": self.classes[cid].name, "color": list(self.classes[cid].color)} for cid in self.class_order]
            self._write_config()
            self._refresh_overlay()

    def _set_current_class_color(self):
        cid = self._current_class_id()
        if cid is None:
            return
        c = self.classes[cid].color
        qc = QColorDialog.getColor(QColor(c[0], c[1], c[2]), self, "选择颜色")
        if not qc.isValid():
            return
        self.classes[cid].color = (qc.red(), qc.green(), qc.blue(), c[3])
        self.cfg["classes"] = [{"id": self.classes[cid].id, "name": self.classes[cid].name, "color": list(self.classes[cid].color)} for cid in self.class_order]
        self._write_config()
        self._refresh_overlay()

    def _current_class_id(self) -> Optional[int]:
        idx = self.cmb_class.currentIndex()
        return self.cmb_class.currentData() if idx >= 0 else None

    # ---------------------- 工具状态 ----------------------
    def _on_tool_changed(self):
        name = self.cmb_tool.currentText().lower()
        if "prompt" in name: self.view.set_mode("prompt")
        elif "box" in name: self.view.set_mode("box")
        elif "brush" in name: self.view.set_mode("brush")
        elif "eraser" in name: self.view.set_mode("eraser")
        elif "lasso" in name: self.view.set_mode("lasso")
        elif "smart" in name: self.view.set_mode("smart")
        else: self.view.set_mode("prompt")

    # ---------------------- Prompt 点 ----------------------
    def _on_prompt_click(self, pos_xy: Tuple[float,float], label: int):
        doc = self._cur_doc()
        if doc is None or doc.size_hw == (0,0):
            return
        x, y = int(round(pos_xy[0])), int(round(pos_xy[1]))
        h, w = doc.size_hw
        if not (0 <= x < w and 0 <= y < h):
            return
        r = 4
        item = QGraphicsEllipseItem(x-r, y-r, r*2, r*2)
        color = QColor(0,255,0,255) if label==1 else QColor(255,0,0,255)
        item.setBrush(QBrush(color)); item.setPen(QPen(QColor(0,0,0,200), 1)); item.setZValue(30)
        self.scene.addItem(item)
        self.clicks.append(ClickPoint((x,y), label, item))
        self._update_prompt_labels()

    def _delete_nearest_point(self, pos_xy: Tuple[float,float]):
        if not self.clicks:
            return
        sx, sy = pos_xy
        # 按欧氏距离找最近
        best = None; best_d = 1e18
        for i,c in enumerate(self.clicks):
            d = (c.pos[0]-sx)**2 + (c.pos[1]-sy)**2
            if d < best_d: best_d = d; best = i
        if best is None:
            return
        item = self.clicks[best].item
        if item is not None: self.scene.removeItem(item)
        self.clicks.pop(best)
        self._update_prompt_labels()

    def _update_prompt_labels(self):
        pos_cnt = sum(1 for c in self.clicks if c.label==1)
        neg_cnt = sum(1 for c in self.clicks if c.label==0)
        self.lbl_pts.setText(f"Points: {pos_cnt}(+)/{neg_cnt}(-)")
        self.lbl_boxes.setText(f"Boxes: {len(self.boxes)}")

    # ---------------------- Box ----------------------
    def _on_box_update(self, box_xyxy: Tuple[float,float,float,float]):
        x0, y0, x1, y1 = box_xyxy
        x_min, x_max = sorted([x0, x1]); y_min, y_max = sorted([y0, y1])
        if self.box_preview is None:
            self.box_preview = QGraphicsRectItem()
            self.box_preview.setZValue(20)
            self.box_preview.setPen(QPen(QColor(255, 255, 255, 200), 2, Qt.DashLine))
            self.box_preview.setBrush(QBrush(QColor(0, 120, 255, 40)))
            self.scene.addItem(self.box_preview)
        self.box_preview.setRect(QRectF(x_min, y_min, x_max-x_min, y_max-y_min))
        self.box_preview.setVisible(True)

    def _on_box_finish(self, box_xyxy: Tuple[float,float,float,float]):
        x0, y0, x1, y1 = box_xyxy
        x_min, x_max = sorted([x0, x1]); y_min, y_max = sorted([y0, y1])
        rect = (x_min, y_min, x_max, y_max)
        item = QGraphicsRectItem(QRectF(x_min, y_min, x_max-x_min, y_max-y_min))
        item.setZValue(20)
        item.setPen(QPen(QColor(255, 255, 255, 200), 2, Qt.SolidLine))
        item.setBrush(QBrush(QColor(0, 120, 255, 40)))
        self.scene.addItem(item)
        self.boxes.append(BoxEntry(rect, item))
        # 隐藏预览
        if self.box_preview is not None:
            self.box_preview.setVisible(False)
        self._update_prompt_labels()

    def _delete_box_at(self, pos_xy: Tuple[float,float]):
        if not self.boxes:
            return
        sx, sy = pos_xy
        # 优先删除包含该点的 box
        idx = None
        for i, b in enumerate(self.boxes):
            x0,y0,x1,y1 = b.rect
            if x0 <= sx <= x1 and y0 <= sy <= y1:
                idx = i; break
        if idx is None:
            # 否则删最近中心
            best = None; best_d = 1e18
            for i,b in enumerate(self.boxes):
                x0,y0,x1,y1 = b.rect
                cx, cy = (x0+x1)/2, (y0+y1)/2
                d = (cx-sx)**2 + (cy-sy)**2
                if d < best_d: best_d = d; best = i
            idx = best
        if idx is None:
            return
        item = self.boxes[idx].item
        self.scene.removeItem(item)
        self.boxes.pop(idx)
        self._update_prompt_labels()

    # ---------------------- Brush/Eraser ----------------------
    def _on_brush_draw(self, pos_xy: Tuple[float,float], radius: int, add: bool):
        doc = self._cur_doc()
        if doc is None:
            return
        cid = self._current_class_id();
        if cid is None:
            return
        if cid not in doc.masks or doc.masks[cid] is None:
            doc.masks[cid] = np.zeros(doc.size_hw, dtype=bool)
        x, y = int(round(pos_xy[0])), int(round(pos_xy[1]))
        paint_disk(doc.masks[cid], (y, x), radius, True if add else False)
        doc.dirty = True
        self._refresh_overlay()

    # ---------------------- 智能选取/Lasso ----------------------
    def _on_smart_click(self, pos_xy: Tuple[float,float], tol: int):
        doc = self._cur_doc()
        if doc is None or doc.image_rgb is None:
            return
        cid = self._current_class_id()
        if cid is None:
            return
        x, y = int(round(pos_xy[0])), int(round(pos_xy[1]))
        h, w = doc.size_hw
        if not (0 <= x < w and 0 <= y < h):
            return
        self._begin_mask_edit()
        mask = smart_select(doc.image_rgb, (x,y), tol)
        if cid not in doc.masks or doc.masks[cid] is None:
            doc.masks[cid] = np.zeros(doc.size_hw, dtype=bool)
        doc.masks[cid] |= mask
        doc.dirty = True
        self._refresh_overlay()
        self._end_mask_edit("SmartSelect")

    def _on_lasso_update(self, pts: List[Tuple[float,float]]):
        # 如需实时显示路径，可在此添加一个 QGraphicsPathItem（此处略）
        """
            pts: list of (x, y) scene coordinates
            在用户拖动时被调用，实时更新一条 QGraphicsPathItem
            """
        # remove previous if exists? better reuse
        if not pts:
            if self.lasso_path_item is not None:
                self.scene.removeItem(self.lasso_path_item)
                self.lasso_path_item = None
            return

        # prepare path
        path = QPainterPath()
        path.moveTo(pts[0][0], pts[0][1])
        for (x, y) in pts[1:]:
            path.lineTo(x, y)
        path.closeSubpath()

        if self.lasso_path_item is None:
            self.lasso_path_item = QGraphicsPathItem()
            self.lasso_path_item.setZValue(45)  # 高于 overlay/candidate
            # choose pen/brush based on current class color (fallback to white)
            cid = self._current_class_id()
            if cid is not None:
                R, G, B, A = self.classes[cid].color
                pen = QPen(QColor(R, G, B, 200), 2)
                brush = QBrush(QColor(R, G, B, 50))
            else:
                pen = QPen(QColor(255, 255, 255, 200), 2)
                brush = QBrush(QColor(255, 255, 255, 40))
            self.lasso_path_item.setPen(pen)
            self.lasso_path_item.setBrush(brush)
            self.scene.addItem(self.lasso_path_item)

        self.lasso_path_item.setPath(path)
        self.lasso_path_item.setVisible(True)

    def _on_lasso_finish(self, pts: List[Tuple[float,float]]):
        """
            抽取多边形 pts 为 mask 并应用到当前类别，
            并使用 begin/end 以支持撤销/重做。完成后删除临时 path。
            """
        doc = self._cur_doc()
        if doc is None:
            # remove preview if any
            if self.lasso_path_item is not None:
                self.scene.removeItem(self.lasso_path_item)
                self.lasso_path_item = None
            return

        if len(pts) < 3:
            # too few pts: just clear preview
            if self.lasso_path_item is not None:
                self.scene.removeItem(self.lasso_path_item)
                self.lasso_path_item = None
            return

        cid = self._current_class_id()
        if cid is None:
            QMessageBox.warning(self, "Lasso", "请选择一个类别后再使用套索工具。")
            # clean preview
            if self.lasso_path_item is not None:
                self.scene.removeItem(self.lasso_path_item)
                self.lasso_path_item = None
            return
        # remove preview path
        if self.lasso_path_item is not None:
            # optionally keep path as permanent outline? we remove to avoid clutter
            self.scene.removeItem(self.lasso_path_item)
            self.lasso_path_item = None
        # begin undo snapshot
        self._begin_mask_edit()

        # rasterize polygon to mask aligned with image pixels
        poly_mask = rasterize_polygon_mask(pts, doc.size_hw)
        if cid not in doc.masks or doc.masks[cid] is None:
            doc.masks[cid] = np.zeros(doc.size_hw, dtype=bool)
        doc.masks[cid] |= poly_mask
        doc.dirty = True

        # refresh overlay to show result
        self._refresh_overlay()

        # end undo snapshot
        self._end_mask_edit("Lasso")


        # doc = self._cur_doc()
        # if doc is None:
        #     return
        # cid = self._current_class_id()
        # if cid is None:
        #     return
        # self._begin_mask_edit()
        # poly_mask = rasterize_polygon_mask(pts, doc.size_hw)
        # if cid not in doc.masks or doc.masks[cid] is None:
        #     doc.masks[cid] = np.zeros(doc.size_hw, dtype=bool)
        # doc.masks[cid] |= poly_mask
        # doc.dirty = True
        # self._refresh_overlay()
        # self._end_mask_edit("Lasso")

    # ---------------------- SAM 运行/应用 ----------------------
    def _run_sam(self):
        doc = self._cur_doc()
        if doc is None or doc.image_rgb is None:
            QMessageBox.warning(self, "SAM", "请先选择并加载一张图片。"); return
        if self.predictor.backend is None:
            QMessageBox.warning(self, "SAM", "SAM 后端不可用，请在“设置→设置模型”中配置。"); return
        pts = None; lbls = None
        if len(self.clicks) > 0:
            pts = np.array([c.pos for c in self.clicks], dtype=np.float32)
            lbls = np.array([c.label for c in self.clicks], dtype=np.int32)
        box = None
        if self.boxes:
            box = np.array(self.boxes[-1].rect, dtype=np.float32)  # 使用最后一个框
        self.predictor.set_image(doc.image_rgb)
        mask = self.predictor.predict(pts, lbls, box)
        if mask is None:
            QMessageBox.warning(self, "SAM", "未返回有效 mask，请添加更多提示或调整框。"); return
        self.candidate_mask = mask
        self._refresh_candidate()

    def _apply_candidate_to_class(self):
        doc = self._cur_doc()
        if doc is None or self.candidate_mask is None:
            return
        cid = self._current_class_id()
        if cid is None:
            return
        if cid not in doc.masks or doc.masks[cid] is None:
            doc.masks[cid] = np.zeros(doc.size_hw, dtype=bool)
        self._begin_mask_edit()
        doc.masks[cid] |= self.candidate_mask
        for k in list(doc.masks.keys()):
            if k != cid and doc.masks.get(k) is not None:
                doc.masks[k] &= ~self.candidate_mask
        doc.dirty = True
        self._refresh_overlay()
        self._end_mask_edit("Apply Candidate")

    def _refresh_candidate(self):
        if self.candidate_mask is None:
            self.candidate_item.setPixmap(QPixmap()); return
        img = colorize_mask(self.candidate_mask, (255,255,255,100))
        self.candidate_item.setPixmap(QPixmap.fromImage(img))

    def _clear_prompts(self):
        # 清点
        for c in self.clicks:
            if c.item is not None: self.scene.removeItem(c.item)
        self.clicks.clear()
        # 清框
        for b in self.boxes:
            self.scene.removeItem(b.item)
        self.boxes.clear()
        if self.box_preview is not None:
            self.box_preview.setVisible(False)
        if hasattr(self, "lasso_path_item") and self.lasso_path_item is not None:
            print(">>>>>>>>>>>>")
            self.scene.removeItem(self.lasso_path_item)
            self.lasso_path_item = None
        # 清候选
        self.candidate_mask = None
        self.candidate_item.setPixmap(QPixmap())
        self._update_prompt_labels()

    # ---------------------- 多文件/加载保存 ----------------------
    def _add_files(self):
        fns, _ = QFileDialog.getOpenFileNames(self, "添加图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not fns:
            return
        for fn in fns:
            self.docs.append(ImageDoc(fn))
            self.list_files.addItem(QListWidgetItem(os.path.basename(fn)))
        if self.cur_index < 0 and self.docs:
            self.list_files.setCurrentRow(0)

    def _add_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择文件夹", "")
        if not d:
            return
        exts = (".png",".jpg",".jpeg",".bmp")
        files = [str(p) for p in pathlib.Path(d).glob("**/*") if p.suffix.lower() in exts]
        files.sort()
        for fn in files:
            self.docs.append(ImageDoc(fn))
            self.list_files.addItem(QListWidgetItem(os.path.basename(fn)))
        if self.cur_index < 0 and self.docs:
            self.list_files.setCurrentRow(0)

    def _on_select_index(self, idx: int):
        if idx == self.cur_index:
            return
        # 切换前自动保存
        if self.cur_index >= 0 and self.cfg.get("auto_save", True):
            self._export_current(auto=True)
        self.cur_index = idx
        if idx < 0 or idx >= len(self.docs):
            self._clear_canvas()
            return
        doc = self.docs[idx]
        if doc.image_rgb is None or doc.base_pixmap is None:
            try:
                img = Image.open(doc.path).convert("RGB")
                doc.image_rgb = np.array(img)
                h, w = doc.image_rgb.shape[:2]
                doc.size_hw = (h, w)
                doc.base_pixmap = QPixmap.fromImage(qimage_from_rgb(doc.image_rgb))
                self._maybe_load_existing_masks(doc)
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"{doc.path}\n{e}"); return
        self._clear_prompts()
        self._set_canvas_doc(doc)

    def _clear_canvas(self):
        self.base_item.setPixmap(QPixmap()); self.overlay_item.setPixmap(QPixmap())
        self.candidate_item.setPixmap(QPixmap()); self.scene.setSceneRect(QRectF(0,0,0,0))

    def _set_canvas_doc(self, doc: ImageDoc):
        self.base_item.setPixmap(doc.base_pixmap)
        h, w = doc.size_hw
        self.scene.setSceneRect(QRectF(0,0,w,h))
        self._refresh_overlay()

    def _export_current(self, auto: bool=False):
        doc = self._cur_doc()
        if doc is None:
            return
        save_dir = self.cfg.get("save_dir", "")
        if not save_dir:
            base = os.path.splitext(doc.path)[0]
            save_dir = base
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(doc.path))[0]
        out_dir = os.path.join(save_dir, base_name)
        os.makedirs(out_dir, exist_ok=True)
        h, w = doc.size_hw
        label_map = np.zeros((h, w), dtype=np.uint8)
        for cid in self.class_order:
            m = doc.masks.get(cid, None)
            if m is None: m = np.zeros((h, w), dtype=bool)
            Image.fromarray((m.astype(np.uint8)*255)).save(os.path.join(out_dir, f"{cid}_{self.classes[cid].name}.png"))
            label_map[m] = cid
        Image.fromarray(label_map).save(os.path.join(out_dir, "labels.png"))
        meta = {
            "image": doc.path,
            "classes": [{"id": self.classes[cid].id, "name": self.classes[cid].name, "color": list(self.classes[cid].color)} for cid in self.class_order]
        }
        with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        doc.dirty = False
        if not auto:
            QMessageBox.information(self, "导出完毕", f"已保存到：\n{out_dir}")

    def _maybe_load_existing_masks(self, doc: ImageDoc):
        base_name = os.path.splitext(os.path.basename(doc.path))[0]
        base_dir = self.cfg.get("save_dir", "") or os.path.dirname(doc.path)
        out_dir = os.path.join(os.path.abspath(base_dir), base_name)
        if not os.path.isdir(out_dir):
            alt = os.path.join(os.path.dirname(doc.path), base_name)
            if os.path.isdir(alt):
                out_dir = alt
            else:
                return
        h, w = doc.size_hw
        labels_png = os.path.join(out_dir, "labels.png")
        if os.path.isfile(labels_png):
            arr = np.array(Image.open(labels_png).convert("L"))
            for cid in self.class_order:
                doc.masks[cid] = (arr == cid)
            doc.dirty = False; return
        for cid in self.class_order:
            fn = os.path.join(out_dir, f"{cid}_{self.classes[cid].name}.png")
            if os.path.isfile(fn):
                arr = np.array(Image.open(fn).convert("L"))
                doc.masks[cid] = (arr > 127)
        doc.dirty = False

    # ---------------------- 模型/保存设置 ----------------------
    def _set_model(self):
        types = ["sam2", "sam"]
        model_type, ok = QInputDialog.getItem(self, "模型类型", "Type:", types, editable=False)
        if not ok:
            return
        cfg_path = ""
        ckpt_path = ""
        if model_type == "sam2":
            cfg_path, _ = QFileDialog.getOpenFileName(self, "选择 SAM2 配置文件（.yaml）", "", "YAML (*.yaml *.yml);;All Files (*.*)")
            if not cfg_path:
                QMessageBox.warning(self, "SAM2", "必须选择配置文件（model_cfg）。")
                return
            ckpt_path, _ = QFileDialog.getOpenFileName(self, "选择 SAM2 权重（.pth/.pt）", "", "Weights (*.pth *.pt);;All Files (*.*)")
            if not ckpt_path:
                QMessageBox.warning(self, "SAM2", "必须选择权重文件（model_ckpt）。")
                return
        else:
            ckpt_path, _ = QFileDialog.getOpenFileName(self, "选择 SAM(v1) 权重（.pth）", "", "Weights (*.pth *.pt);;All Files (*.*)")
            if not ckpt_path:
                QMessageBox.warning(self, "SAM", "必须选择权重文件。")
                return

        self.cfg["model_type"] = model_type
        self.cfg["model_cfg"] = cfg_path
        self.cfg["model_ckpt"] = ckpt_path
        self._write_config()
        self.predictor = PredictorAdapter(model_type=model_type, checkpoint_path=ckpt_path or None, model_cfg=cfg_path or None)
        QMessageBox.information(self, "模型", f"模型：{model_type}，后端：{'可用' if self.predictor.backend else '不可用'}")

    def _set_save_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择保存路径", "")
        if not d:
            return
        self.cfg["save_dir"] = d
        self._write_config()

    def _toggle_auto_save(self, v: bool):
        self.cfg["auto_save"] = bool(v)
        self._write_config()

    # ---------------------- 覆盖层刷新 ----------------------
    def _refresh_overlay(self):
        doc = self._cur_doc()
        if doc is None or doc.size_hw == (0,0):
            self.overlay_item.setPixmap(QPixmap()); self.candidate_item.setPixmap(QPixmap()); return
        h, w = doc.size_hw
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        alpha = int(self.slider_alpha.value())
        for cid in self.class_order:
            m = doc.masks.get(cid, None)
            if m is None: continue
            R, G, B, _ = self.classes[cid].color
            a = max(0, min(255, alpha))
            rgba[m, 0] = R; rgba[m, 1] = G; rgba[m, 2] = B; rgba[m, 3] = a
        qimg = QImage(rgba.data, w, h, 4*w, QImage.Format_RGBA8888).copy()
        self.overlay_item.setPixmap(QPixmap.fromImage(qimg))
        self._refresh_candidate()

    # ---------------------- 撤销/重做（修复版） ----------------------
    class MaskEditCommand(QUndoCommand):
        def __init__(self, win: "MainWindow", before: Dict[int, np.ndarray], after: Dict[int, np.ndarray], text: str = "mask edit"):
            super().__init__(text)
            self.w = win
            self.before = {cid: (m.copy() if m is not None else None) for cid,m in before.items()}
            self.after  = {cid: (m.copy() if m is not None else None) for cid,m in after.items()}

        def _apply(self, data: Dict[int, np.ndarray]):
            d = self.w._cur_doc()
            if d is None:
                return
            # 覆盖整个 masks 字典（按现有类别 id）
            d.masks = {cid: (m.copy() if m is not None else None) for cid,m in data.items()}
            d.dirty = True
            self.w._refresh_overlay()

        def undo(self):
            self._apply(self.before)

        def redo(self):
            self._apply(self.after)

    def _begin_mask_edit(self):
        doc = self._cur_doc()
        if doc is None:
            return
        if self._capture_before is None:
            self._capture_before = {cid: (m.copy() if m is not None else None) for cid,m in doc.masks.items()}

    def _end_mask_edit(self, text="mask edit"):
        doc = self._cur_doc()
        if doc is None:
            self._capture_before = None; return
        if self._capture_before is None:
            return
        after = {cid: (m.copy() if m is not None else None) for cid,m in doc.masks.items()}
        before = self._capture_before
        self._capture_before = None
        self.undo_stack.push(MainWindow.MaskEditCommand(self, before, after, text))

    # ---------------------- 批处理 ----------------------
    def _apply_sam_to_all(self):
        if self.predictor.backend is None:
            QMessageBox.warning(self, "SAM", "SAM 后端不可用。"); return
        if not self.docs:
            return
        if len(self.clicks)==0 and not self.boxes:
            QMessageBox.warning(self, "SAM", "请先添加 Prompt 点或框并 Run SAM。")
            return
        cid = self._current_class_id()
        if cid is None:
            QMessageBox.warning(self, "SAM", "请选择要应用的类别。")
            return
        pts = np.array([c.pos for c in self.clicks], dtype=np.float32) if self.clicks else None
        lbls = np.array([c.label for c in self.clicks], dtype=np.int32) if self.clicks else None
        box = np.array(self.boxes[-1].rect, dtype=np.float32) if self.boxes else None

        for i, doc in enumerate(self.docs):
            if doc.image_rgb is None:
                try:
                    img = Image.open(doc.path).convert("RGB")
                    doc.image_rgb = np.array(img)
                    h, w = doc.image_rgb.shape[:2]
                    doc.size_hw = (h, w)
                except Exception as e:
                    print("加载失败：", doc.path, e); continue
            self.predictor.set_image(doc.image_rgb)
            mask = self.predictor.predict(pts, lbls, box)
            if mask is None:
                print("SAM 无结果：", doc.path); continue
            if cid not in doc.masks or doc.masks[cid] is None:
                doc.masks[cid] = np.zeros(doc.size_hw, dtype=bool)
            doc.masks[cid] |= mask
            for k in list(doc.masks.keys()):
                if k != cid and doc.masks.get(k) is not None:
                    doc.masks[k] &= ~mask
            doc.dirty = True
            self.cur_index = i
            self._export_current(auto=True)
        QMessageBox.information(self, "批处理完成", "已对所有图像应用当前 SAM 提示并保存。")

    # ---------------------- 其它 ----------------------
    def _cur_doc(self) -> Optional[ImageDoc]:
        return self.docs[self.cur_index] if 0 <= self.cur_index < len(self.docs) else None

    def _refresh_ui_state(self):
        self._refresh_class_combo(); self._on_tool_changed()

    def closeEvent(self, event):
        try:
            if self.cfg.get("auto_save", True):
                self._export_current(auto=True)
        except Exception:
            pass
        self._write_config()
        super().closeEvent(event)

# ----------------------------- 入口 -----------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

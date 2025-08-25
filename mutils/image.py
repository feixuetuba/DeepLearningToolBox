import cv2
from PySide6 import QtGui
import numpy as np

def cvbgr_to_qimage(cv_bgr: np.ndarray) -> QtGui.QImage:
    """Convert BGR(A) OpenCV image to QImage (RGB or RGBA)"""
    if cv_bgr is None:
        return QtGui.QImage()
    h, w = cv_bgr.shape[:2]
    if cv_bgr.ndim == 2:
        fmt = QtGui.QImage.Format_Grayscale8
        img = QtGui.QImage(cv_bgr.data, w, h, cv_bgr.strides[0], fmt)
        return img.copy()
    # color
    if cv_bgr.shape[2] == 3:
        rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        return img.copy()
    else:
        rgba = cv2.cvtColor(cv_bgr, cv2.COLOR_BGRA2RGBA)
        img = QtGui.QImage(rgba.data, w, h, rgba.strides[0], QtGui.QImage.Format_RGBA8888)
        return img.copy()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier-Transform Drawing (epicycles) — PySide6 + numpy. Single file.

YouTube / Shorts edition
------------------------
* Platform-framed export: Shorts 9:16 (1080×1920), YouTube 16:9 (1920×1080 /
  1280×720 / 2560×1440 / 3840×2160), Square 1:1, or custom. The artwork is
  laid out inside a "stage" that respects YouTube/Shorts UI safe zones —
  title on top, captions + progress bar below, art never clipped.
* Baked title-card intro (editable title / subtitle / @handle) and a
  Subscribe-pill outro — the video is upload-ready with zero editing.
* Live captions while drawing ("stroke 12 / 48 · 2,600 epicycles"),
  thin progress bar, channel watermark — each toggleable.
* In-dialog live preview of the composed frame (intro / mid-draw / outro).
* Quality presets (CRF), ETA during export, optional 2× supersampling,
  same backends: imageio-ffmpeg, system ffmpeg pipe, GIF, PNG sequence.
* Everything from before still works: marching-squares paths, tone contours,
  neon bloom, rainbow ink, sparkles, scrubbing, PNG export, shortcuts.

Deps:  pip install PySide6 numpy
       MP4:  pip install imageio imageio-ffmpeg   (or system ffmpeg on PATH)
       GIF:  pip install pillow
"""
from collections import defaultdict
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import numpy as np

try:
    import imageio                      # noqa: N813
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False

try:
    from PIL import Image
    _HAS_PIL = True
    try:
        _GIF_DITHER = Image.Dither.FLOYDSTEINBERG
    except AttributeError:
        _GIF_DITHER = Image.FLOYDSTEINBERG
except Exception:
    Image = None
    _HAS_PIL = False

from PySide6.QtCore import (Qt, QTimer, Signal, QObject, QRectF, QPointF,
                            QSettings)
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QImage, QPixmap,
                           QPolygonF, QShortcut, QKeySequence, QLinearGradient,
                           QRadialGradient, QFont, QFontMetrics)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QSlider,
                               QCheckBox, QGroupBox, QFileDialog, QMessageBox,
                               QSizePolicy, QComboBox, QScrollArea, QDialog,
                               QProgressBar, QSpinBox, QDoubleSpinBox,
                               QGridLayout, QLineEdit)

PRESETS = {
    "Mona Lisa": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/"
        "Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/"
        "800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        "Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg?width=800",
    ],
    "The Starry Night": [
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg?width=1024",
    ],
}

STYLESHEET = """
QMainWindow, QWidget, QDialog { background-color: #10141d; color: #dfe7f1;
                                font-size: 13px; }
#title  { font-size: 16px; font-weight: 600; color: #ffffff; }
#val    { color: #8ab4f8; font-weight: 600; }
#status { color: #9fb3c8; }
#info   { color: #7d8fa3; font-size: 12px; }
#preview { border: 1px solid #26303f; border-radius: 8px; background: #0b0f18; }
QPushButton { background-color: qlineargradient(x1:0,y1:0,x2:0,y2:1,
               stop:0 #263852, stop:1 #1f2e46);
              border: 1px solid #3a5378; border-radius: 7px; padding: 6px 10px; }
QPushButton:hover  { background-color: #2e4463; border-color: #4f6f9e; }
QPushButton:pressed{ background-color: #1a2740; }
QPushButton:disabled { color: #66788e; background-color: #1a2230; border-color: #26324a; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #223047; border: 1px solid #3a5378;
    border-radius: 7px; padding: 4px 8px; }
QComboBox QAbstractItemView { background-color: #1b2433;
                              selection-background-color: #2e4463; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 2px; }
QScrollBar::handle:vertical { background: #2c3a52; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #3a4c6c; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QToolTip { background-color: #1b2433; color: #dfe7f1; border: 1px solid #3a5378;
           padding: 4px; }
QSlider::groove:horizontal { height: 4px; background: #26324a; border-radius: 2px; }
QSlider::sub-page:horizontal { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
               stop:0 #3f6ad8, stop:1 #7a5cff); border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -6px 0;
                             border-radius: 7px; background: #cfe0ff; }
QSlider::handle:horizontal:hover { background: #ffffff; }
QProgressBar { background: #182236; border: 1px solid #3a5378; border-radius: 6px;
               text-align: center; color: #dfe7f1; min-height: 16px; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
               stop:0 #3f6ad8, stop:1 #7a5cff); border-radius: 5px; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 15px; height: 15px; border-radius: 4px;
                       border: 1px solid #3a5378; background: #182236; }
QCheckBox::indicator:hover { border-color: #5f8bef; }
QCheckBox::indicator:checked { background: qradialgradient(cx:0.5, cy:0.5,
                       radius:0.6, fx:0.5, fy:0.5, stop:0 #9cc3ff, stop:1 #3f6ad8);
                       border-color: #6f9bf0; }
QGroupBox { border: 1px solid #26303f; border-radius: 9px; margin-top: 12px;
            padding: 8px 6px 6px 6px; background: rgba(255,255,255,0.015); }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                   color: #8ab4f8; }
"""


# ===========================================================================
# Contour extraction (pure numpy)
# ===========================================================================
def _conv1(a, k, axis):
    pad = [(0, 0), (0, 0)]
    pad[axis] = (1, 1)
    p = np.pad(a, pad, mode="edge")
    if axis == 0:
        return k[0] * p[:-2, :] + k[1] * p[1:-1, :] + k[2] * p[2:, :]
    return k[0] * p[:, :-2] + k[1] * p[:, 1:-1] + k[2] * p[:, 2:]


def marching_squares(field, level):
    H, W = field.shape
    F = np.pad(field, 1, mode="constant", constant_values=float(level) - 1.0)
    a = F[:-1, :-1]; b = F[:-1, 1:]; c = F[1:, 1:]; d = F[1:, :-1]
    case = (((a > level).astype(np.uint8) << 3)
            | ((b > level).astype(np.uint8) << 2)
            | ((c > level).astype(np.uint8) << 1)
            | (d > level).astype(np.uint8))
    rs, cs = np.nonzero((case != 0) & (case != 15))
    if rs.size == 0:
        return []

    def fr(p, q, lv=float(level)):
        dd = q - p
        return 0.5 if -1e-12 < dd < 1e-12 else (lv - p) / dd

    segs = []
    for r, col, k, A, B, C, D in zip(rs.tolist(), cs.tolist(),
                                     case[rs, cs].tolist(),
                                     F[rs, cs].tolist(), F[rs, cs + 1].tolist(),
                                     F[rs + 1, cs + 1].tolist(), F[rs + 1, cs].tolist()):
        x0, y0 = float(col), float(r)
        T = (x0 + fr(A, B), y0)
        R = (x0 + 1.0, y0 + fr(B, C))
        Bm = (x0 + fr(D, C), y0 + 1.0)
        L = (x0, y0 + fr(A, D))
        if k == 1:    segs.append((L, Bm))
        elif k == 2:  segs.append((Bm, R))
        elif k == 3:  segs.append((L, R))
        elif k == 4:  segs.append((T, R))
        elif k == 5:  segs += [(T, L), (Bm, R)] if (A + B + C + D) * .25 > level \
                            else [(T, R), (L, Bm)]
        elif k == 6:  segs.append((T, Bm))
        elif k == 7:  segs.append((T, L))
        elif k == 8:  segs.append((T, L))
        elif k == 9:  segs.append((T, Bm))
        elif k == 10: segs += [(T, R), (L, Bm)] if (A + B + C + D) * .25 > level \
                            else [(T, L), (Bm, R)]
        elif k == 11: segs.append((T, R))
        elif k == 12: segs.append((L, R))
        elif k == 13: segs.append((Bm, R))
        elif k == 14: segs.append((L, Bm))
    return _assemble_polylines(segs)


def _assemble_polylines(segs):
    if not segs:
        return []
    coord = {}

    def key(p):
        k = (round(p[0], 4), round(p[1], 4))
        if k not in coord:
            coord[k] = p
        return k

    edges, nbrs = [], defaultdict(list)
    for i, (p, q) in enumerate(segs):
        kp, kq = key(p), key(q)
        edges.append((kp, kq))
        nbrs[kp].append(i)
        nbrs[kq].append(i)

    used = [False] * len(segs)
    lines = []
    for i0 in range(len(segs)):
        if used[i0]:
            continue
        used[i0] = True
        ka, kb = edges[i0]
        chain = [ka, kb]
        closed = False
        cur, start = kb, ka
        while True:
            nxt = -1
            for s in nbrs[cur]:
                if not used[s]:
                    nxt = s
                    break
            if nxt < 0:
                break
            used[nxt] = True
            pa, pb = edges[nxt]
            cur = pb if pa == cur else pa
            if cur == start:
                closed = True
                break
            chain.append(cur)
        if closed:
            chain.append(start)
        else:
            pre = []
            cur = ka
            while True:
                nxt = -1
                for s in nbrs[cur]:
                    if not used[s]:
                        nxt = s
                        break
                if nxt < 0:
                    break
                used[nxt] = True
                pa, pb = edges[nxt]
                cur = pb if pa == cur else pa
                pre.append(cur)
            pre.reverse()
            chain = pre + chain
        lines.append(np.array([coord[k] for k in chain], dtype=np.float64))
    return lines


def _poly_len(pts):
    d = np.roll(pts, -1, axis=0) - pts
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def resample_polyline(pts, n):
    closed = len(pts) >= 2 and np.allclose(pts[0], pts[-1], atol=1e-6)
    if closed:
        pts = pts[:-1]
    n = max(int(n), 3)
    if closed:
        seg = np.roll(pts, -1, axis=0) - pts
    else:
        seg = np.diff(pts, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    s = np.linspace(0.0, total, n, endpoint=False) if closed \
        else np.linspace(0.0, total, n)
    idx = np.searchsorted(cum, s, side="right") - 1
    idx = np.clip(idx, 0, len(seglen) - 1)
    frac = (s - cum[idx]) / np.maximum(seglen[idx], 1e-12)
    return pts[idx] + seg[idx] * frac[:, None]


def _circ_smooth(pts, k):
    n = len(pts)
    if k <= 0 or n < 2 * k + 3:
        return pts
    ext = np.vstack([pts[-k:], pts, pts[:k]])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.column_stack([np.convolve(ext[:, 0], ker, "valid"),
                            np.convolve(ext[:, 1], ker, "valid")])


def extract_contour_paths(gray, detail, sensitivity, mode):
    k1 = np.array([1.0, 2.0, 1.0]) / 4.0
    sx = np.array([-1.0, 0.0, 1.0])
    b1 = _conv1(_conv1(gray, k1, 0), k1, 1)
    b2 = _conv1(_conv1(b1, k1, 0), k1, 1)

    fields, min_len = [], 6.0
    if mode == "tone":
        b3 = _conv1(_conv1(b2, k1, 0), k1, 1)
        seen = set()
        for q in (0.86, 0.72, 0.58, 0.44, 0.30, 0.16):
            lv = float(np.quantile(b3, q))
            if round(lv, 6) not in seen:
                seen.add(round(lv, 6))
                fields.append((b3, lv))
        min_len = 10.0
    else:
        gx = _conv1(_conv1(b2, sx, 1), k1, 0)
        gy = _conv1(_conv1(b2, k1, 1), sx, 0)
        mag = np.hypot(gx, gy)
        q = 0.9995 - (sensitivity / 100.0) * 0.05
        fields.append((mag, max(float(np.quantile(mag, q)), 6.0)))

    polys = []
    for f, lv in fields:
        for pts in marching_squares(f, lv):
            pts = pts - 1.0
            L = _poly_len(pts)
            if L >= min_len and len(pts) >= 4:
                polys.append((pts, L))
    if not polys:
        raise ValueError("No contours found — try a different mode or image.")
    polys.sort(key=lambda t: -t[1])
    polys = polys[:90]

    total = sum(L for _, L in polys)
    out = []
    for pts, L in polys:
        n = int(np.clip(round(detail * L / total), 28, 1600))
        r = _circ_smooth(resample_polyline(pts, n), 2 if n >= 80 else 1)
        out.append(np.ascontiguousarray(r))
    return out


# ===========================================================================
# Fourier engine (multi-path, dense-curve cache)
# ===========================================================================
class FourierPath:
    __slots__ = ("pts", "n", "closed", "coeffs", "w", "amp", "_cache")

    def __init__(self, pts):
        self.pts = np.ascontiguousarray(pts, dtype=np.float64)
        self.n = len(pts)
        self.closed = bool(np.allclose(pts[0], pts[-1], atol=1e-6))
        z = self.pts[:, 0] + 1j * self.pts[:, 1]
        spec = np.fft.fft(z) / self.n
        omega = np.round(np.fft.fftfreq(self.n) * self.n).astype(np.int64)
        order = np.argsort(-np.abs(spec), kind="stable")
        self.coeffs = np.ascontiguousarray(spec[order])
        self.w = 2.0 * np.pi * omega[order]
        self.amp = np.abs(self.coeffs)
        self._cache = {}

    def evaluate(self, ts, m):
        ts = np.atleast_1d(np.asarray(ts, np.float64))
        m = int(min(max(m, 0), self.n))
        out = np.empty(ts.shape, np.complex128)
        if m == 0 or ts.size == 0:
            return out
        for i in range(0, ts.size, 256):
            blk = ts[i:i + 256]
            out[i:i + 256] = (np.exp(1j * blk[:, None] * self.w[None, :m])
                              @ self.coeffs[:m])
        return out

    def dense_curve(self, m):
        m = int(min(max(m, 0), self.n))
        c = self._cache.get(m)
        if c is None:
            if len(self._cache) > 12:
                self._cache.clear()
            npts = int(min(self.n * 2, 1400))
            ts = np.linspace(0.0, 1.0, npts, endpoint=not self.closed)
            c = self.evaluate(ts, m)
            self._cache[m] = c
        return c

    def evaluate_point(self, t, m):
        m = int(min(max(m, 0), self.n))
        if m == 0:
            return 0j
        return complex(np.dot(self.coeffs[:m], np.exp(1j * self.w[:m] * t)))

    def tip_chain(self, t, m):
        m = int(min(max(m, 0), self.n))
        if m == 0:
            return np.zeros(0, np.complex128)
        return np.cumsum(self.coeffs[:m] * np.exp(1j * self.w[:m] * t))


class FourierModel:
    JUMP_FRAC = 0.012

    def __init__(self, paths, size):
        self.paths = paths
        self.size = float(size)
        self.n_total = sum(p.n for p in paths)
        self.ghost_polys = []
        for p in paths:
            poly = QPolygonF()
            for x, y in p.pts:
                poly.append(QPointF(float(x), float(y)))
            self.ghost_polys.append(poly)

        P = len(paths)
        jump = self.JUMP_FRAC if P > 1 else 0.0
        draw_total = max(1e-6, 1.0 - jump * (P - 1))
        wsum = max(1, self.n_total)
        self.schedule, t = [], 0.0
        for i, p in enumerate(paths):
            dur = draw_total * p.n / wsum
            self.schedule.append(("path", i, t, t + dur))
            t += dur
            if i < P - 1:
                self.schedule.append(("jump", i, t, t + jump))
                t += jump
        self._hint = 0

    def seg_at(self, t):
        s, n = self.schedule, len(self.schedule)
        i = min(self._hint, n - 1)
        while i > 0 and t < s[i][2]:
            i -= 1
        while i < n - 1 and t >= s[i][3]:
            i += 1
        self._hint = i
        return s[i]

    def path_start(self, i):
        x, y = self.paths[i].pts[0]
        return complex(x, y)


def build_model(image, size=480, detail=2600, sensitivity=45, mode="edges"):
    if image.isNull():
        raise ValueError("Empty image.")
    img = image.convertToFormat(QImage.Format_ARGB32)
    scaled = img.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
    canvas = QImage(size, size, QImage.Format_RGB32)
    canvas.fill(Qt.GlobalColor.white)
    p = QPainter(canvas)
    p.drawImage((size - scaled.width()) // 2, (size - scaled.height()) // 2, scaled)
    p.end()

    g = canvas.convertToFormat(QImage.Format_Grayscale8)
    h, w, bpl = g.height(), g.width(), g.bytesPerLine()
    gray = np.frombuffer(g.constBits(), dtype=np.uint8).reshape(h, bpl)[:, :w]
    gray = gray.astype(np.float64)

    arrays = extract_contour_paths(gray, detail, sensitivity, mode)
    paths = [FourierPath(a - size / 2.0) for a in arrays]
    if not paths:
        raise ValueError("Nothing to draw.")
    return FourierModel(paths, size)


def make_demo_image(size=640):
    img = QImage(size, size, QImage.Format_RGB32)
    img.fill(QColor("#ffffff"))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    ink = QColor("#141414")
    pen = QPen(ink, 5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    s = size / 640.0

    def R(x, y, w, h): return QRectF(x * s, y * s, w * s, h * s)
    def L(x1, y1, x2, y2): p.drawLine(QPointF(x1 * s, y1 * s), QPointF(x2 * s, y2 * s))

    p.setPen(QPen(ink, 4))
    p.drawRect(R(24, 24, 592, 592))
    p.setPen(pen)
    p.drawArc(R(120, 520, 400, 200), 0, 180 * 16)
    L(292, 430, 292, 540)
    L(348, 430, 348, 540)
    p.drawEllipse(R(190, 110, 260, 320))
    p.drawArc(R(150, 80, 340, 260), 10 * 16, 160 * 16)
    L(168, 290, 180, 520)
    L(472, 290, 460, 520)
    p.drawArc(R(240, 205, 60, 26), 0, 180 * 16)
    p.drawArc(R(340, 205, 60, 26), 0, 180 * 16)
    p.drawEllipse(R(248, 240, 44, 26))
    p.drawEllipse(R(348, 240, 44, 26))
    p.setBrush(ink)
    p.drawEllipse(R(262, 246, 15, 15))
    p.drawEllipse(R(362, 246, 15, 15))
    p.setBrush(Qt.BrushStyle.NoBrush)
    L(320, 258, 306, 330)
    L(306, 330, 328, 340)
    p.drawArc(R(268, 318, 104, 64), 200 * 16, 140 * 16)
    p.end()
    return img


class ImageFetcher(QObject):
    fetched = Signal(object)

    def __init__(self, urls, parent=None):
        super().__init__(parent)
        self.urls = list(urls)
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        err, ok = "unknown error", False
        for url in self.urls:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "FourierEpicyclesDemo/5.0"})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = resp.read()
                img = QImage.fromData(data)
                if not img.isNull():
                    self.fetched.emit(img)
                    ok = True
                    break
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        if not ok:
            self.fetched.emit(err)
        self.running = False


def hue_color(h, sat, val, alpha):
    c = QColor()
    c.setHsvF(float(h) % 1.0, sat, val, max(0.0, min(1.0, alpha / 255.0)))
    return c


# ===========================================================================
# Spectrum widget
# ===========================================================================
class SpectrumWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(92)
        self.amps = None
        self.terms = 0

    def set_data(self, amps, terms):
        self.amps = amps
        self.terms = int(terms)
        self.update()

    def set_marker(self, terms):
        self.terms = int(terms)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.setPen(QColor("#26303f"))
        p.setBrush(QColor("#0e141f"))
        p.drawRoundedRect(r, 8, 8)
        p.setPen(QColor("#7d8fa3"))
        p.drawText(r.adjusted(10, 5, -10, 0), Qt.AlignmentFlag.AlignLeft,
                   "amplitude spectrum · active path")
        area = r.adjusted(10, 24, -10, -18)

        for fy in (0.25, 0.5, 0.75):
            y = area.top() + area.height() * fy
            p.setPen(QColor(255, 255, 255, 12))
            p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))

        if self.amps is None or len(self.amps) == 0:
            p.setPen(QColor("#54627a"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no active path")
            return

        n = min(len(self.amps), 160)
        a = np.asarray(self.amps[:n], float)
        mx = max(float(a[0]), 1e-12)
        bw = area.width() / n
        base = area.bottom()

        grad = QLinearGradient(area.x(), 0, area.right(), 0)
        grad.setColorAt(0.0, QColor("#37c8ff"))
        grad.setColorAt(0.55, QColor("#7f6bff"))
        grad.setColorAt(1.0, QColor("#ff5fa2"))
        sel_brush = QBrush(grad)

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            hgt = (a[i] / mx) ** 0.35 * area.height()
            x = area.x() + i * bw
            p.setBrush(sel_brush if i < self.terms else QColor("#222d44"))
            p.drawRect(QRectF(x, base - hgt, max(1.0, bw - 1.2), hgt))

        p.setPen(QColor(255, 255, 255, 34))
        p.drawLine(QPointF(area.left(), base), QPointF(area.right(), base))

        if self.terms > 0:
            xm = area.x() + min(self.terms, n) * bw
            p.setPen(QPen(QColor(255, 92, 122, 70), 3.0))
            p.drawLine(QPointF(xm, area.top()), QPointF(xm, base))
            p.setPen(QPen(QColor("#ff5c7a"), 1.4))
            p.drawLine(QPointF(xm, area.top()), QPointF(xm, base))

        p.setPen(QColor("#5d7186"))
        p.drawText(r.adjusted(10, 0, -10, -4),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                   f"{min(self.terms, len(self.amps)):,} / {len(self.amps):,} terms")


# ===========================================================================
# Canvas — now layout-aware (stage rect) + baked YouTube/Shorts overlays
# ===========================================================================
class FourierCanvas(QWidget):
    timeChanged = Signal(float)
    playingChanged = Signal(bool)
    activePathChanged = Signal(int)

    LOOP_SECONDS = 16.0
    MARGIN = 14

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(340, 340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.model = None
        self.t = 0.0
        self.playing = True
        self.speed = 1.0
        self.terms = 500
        self.loop = True
        self.show_circles = True
        self.show_arms = True
        self.show_ghost = True
        self.show_jumps = True
        self.rainbow = True
        self.bloom = True
        self.sparkles = True
        self.show_stars = True
        self.show_hud = True
        self.paper = False

        # ---- video composition ----
        self.video_mode = False          # aspect-aware stage layout (exports)
        self.show_video_ui = False       # captions + progress bar + watermark
        self.video_title = ""
        self.video_sub = ""
        self.video_handle = ""
        self.intro_card = 0.0            # 0..1 title-card phase
        self.outro_card = 0.0            # 0..1 subscribe phase

        self._done = []
        self._cur = QPolygonF()
        self._cur_hue = None
        self._jump_lines = []
        self._active_idx = -1
        self._in_jump = None
        self._phase = None
        self._pen = None

        self._particles = []
        self._rng = np.random.default_rng(3)
        self._intro = 0.0
        self._elapsed = 0

        rng = np.random.default_rng(7)
        n = 150
        self._stars = np.column_stack([
            rng.random(n), rng.random(n),
            0.4 + 1.1 * rng.random(n),
            22 + 85 * rng.random(n),
            rng.random(n) * 2 * np.pi,
            0.4 + 1.8 * rng.random(n),
        ])

        self._backdrop = None
        self._core = None
        self._glow = None
        self._layers_key = None
        self._layers_dirty = True

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ---------------- layout ----------------
    def _layout(self, w, h):
        """Stage rect + title/caption anchors for the current frame size."""
        S = self.model.size if self.model else 480.0
        if not self.video_mode:
            mg = self.MARGIN
            size = max(10.0, min(w - 2 * mg, h - 2 * mg))
            return dict(stage=QRectF((w - size) / 2, (h - size) / 2, size, size),
                        title_y=0.06 * h, sub_y=0.10 * h)
        if h >= w:                                    # portrait / Shorts
            size = min(w * 0.94, h * 0.58)
            cy = 0.50 * h
            ty, sy = 0.125 * h, 0.170 * h
        elif w > h * 1.2:                             # landscape 16:9
            size = min(h * 0.72, w * 0.46)
            cy = 0.50 * h
            ty, sy = 0.075 * h, 0.115 * h
        else:                                         # square
            size = min(w, h) * 0.70
            cy = 0.46 * h
            ty, sy = 0.060 * h, 0.100 * h
        return dict(stage=QRectF(w / 2 - size / 2, cy - size / 2, size, size),
                    title_y=ty, sub_y=sy)

    # ---------------- public API ----------------
    def set_model(self, model):
        self.model = model
        self._intro = 1.0
        self.set_time(0.0)

    def set_time(self, t):
        self.t = float(min(max(t, 0.0), 1.0))
        self._rebuild_trail()
        self.update()
        self.timeChanged.emit(self.t)

    def set_terms(self, terms):
        self.terms = int(max(0, terms))
        self._rebuild_trail()
        self.update()

    def set_playing(self, on):
        if self.playing != bool(on):
            self.playing = bool(on)
            self.playingChanged.emit(self.playing)
        self.update()

    # ---------------- layers ----------------
    def _init_layers(self):
        dpr = self.devicePixelRatioF()
        w, h = int(self.width() * dpr), int(self.height() * dpr)
        key = (w, h, round(dpr, 2))
        if w < 2 or h < 2:
            return
        if self._core is None or self._layers_key != key:
            self._core = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
            self._core.setDevicePixelRatio(dpr)
            self._glow = QImage(max(2, self.width() // 3),
                                max(2, self.height() // 3),
                                QImage.Format_ARGB32_Premultiplied)
            self._layers_key = key
        self._core.fill(Qt.GlobalColor.transparent)
        self._glow.fill(Qt.GlobalColor.transparent)

    def _ink(self, hue, a):
        if self.paper:
            if hue is None:
                return QColor(32, 30, 38, a)
            return hue_color(hue, 0.60, 0.32, a)
        if hue is None:
            return QColor(255, 236, 180, a)
        return hue_color(hue, 0.45, 1.0, a)

    def _glow_col(self, hue, a):
        if hue is None:
            return QColor(255, 200, 110, a)
        return hue_color(hue, 0.85, 1.0, a)

    def _bake_poly(self, poly, hue):
        if poly is None or poly.size() < 2 or self.model is None:
            return
        lay = self._layout(self.width(), self.height())
        rect = lay["stage"]
        if rect.width() < 4 or self._core is None:
            return
        s = rect.width() / self.model.size

        p = QPainter(self._core)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.translate(rect.center().x(), rect.center().y())
        p.scale(s, s)
        pen = QPen(self._ink(hue, 235))
        pen.setCosmetic(True)
        pen.setWidthF(1.7)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPolyline(poly)
        p.end()

        if self._glow is None:
            return
        g = QPainter(self._glow)
        g.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        g.translate(self._glow.width() / 2.0, self._glow.height() / 2.0)
        g.scale(s / 3.0, s / 3.0)
        for wd, al in ((10, 26), (4, 60)):
            pen = QPen(self._glow_col(hue, al))
            pen.setCosmetic(True)
            pen.setWidthF(wd)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            g.setPen(pen)
            g.drawPolyline(poly)
        g.end()

    def _rebake_layers(self):
        self._init_layers()
        for poly, hue in self._done:
            self._bake_poly(poly, hue)
        self._layers_dirty = False

    # ---------------- trail bookkeeping ----------------
    def _hue_now(self, idx):
        if not self.rainbow:
            return None
        return ((self._elapsed * 0.00004 + idx * 0.13) % 1.0)

    def _reset_artwork(self):
        self._done.clear()
        self._cur = QPolygonF()
        self._jump_lines.clear()
        self._in_jump = None
        self._phase = None
        self._pen = None
        self._cur_hue = None
        self._particles.clear()
        self._layers_dirty = True
        if self._active_idx != -1:
            self._active_idx = -1
            self.activePathChanged.emit(-1)

    def _end_path(self):
        if self._cur.size() >= 2:
            self._bake_poly(self._cur, self._cur_hue)
        self._cur = QPolygonF()

    def _begin_path(self, idx):
        self._end_path()
        self._in_jump = None
        self._phase = 0.0
        self._cur_hue = self._hue_now(idx)
        if idx != self._active_idx:
            self._active_idx = idx
            self.activePathChanged.emit(idx)

    def _rebuild_trail(self):
        self._reset_artwork()
        m = self.model
        if m is None:
            return
        t, last_idx = self.t, None
        for kind, idx, t0, t1 in m.schedule:
            if t0 >= t:
                break
            if kind == "path":
                u_end = min(1.0, (t - t0) / max(t1 - t0, 1e-9))
                if u_end <= 0:
                    continue
                path = m.paths[idx]
                zs = path.dense_curve(self.terms)
                poly = QPolygonF()
                if u_end >= 0.9995:
                    for z in zs:
                        poly.append(QPointF(float(z.real), float(z.imag)))
                    if path.closed:
                        poly.append(QPointF(float(zs[0].real), float(zs[0].imag)))
                else:
                    k = max(2, int(round(u_end * zs.size)))
                    for z in zs[:k]:
                        poly.append(QPointF(float(z.real), float(z.imag)))
                self._done.append((poly, self._hue_now(idx)))
                last_idx = idx
            else:
                a, b = m.path_start(idx), m.path_start(idx + 1)
                self._jump_lines.append((QPointF(a.real, a.imag),
                                         QPointF(b.real, b.imag)))

        self._rebake_layers()

        tt = min(t, 1.0 - 1e-9)
        kind, idx, t0, t1 = m.seg_at(tt)
        if kind == "path":
            self._begin_path(idx)
            path = m.paths[idx]
            u = (tt - t0) / max(t1 - t0, 1e-9)
            self._phase = u
            z = path.evaluate_point(u, self.terms)
            self._pen = QPointF(z.real, z.imag)
            self._cur.append(self._pen)
        else:
            if last_idx is not None:
                self._begin_path(last_idx)
            self._phase = None
            a, b = m.path_start(idx), m.path_start(idx + 1)
            v = (tt - t0) / max(t1 - t0, 1e-9)
            self._pen = QPointF(a.real + (b.real - a.real) * v,
                                a.imag + (b.imag - a.imag) * v)
            self._in_jump = (QPointF(a.real, a.imag), QPointF(b.real, b.imag))

    # ---------------- particles / fx ----------------
    def _spawn(self, z):
        rng = self._rng
        if len(self._particles) >= 240:
            self._particles.pop(0)
        a = rng.random() * 2 * np.pi
        sp = 4 + 14 * rng.random()
        life = 0.45 + 0.5 * rng.random()
        self._particles.append([z.real, z.imag, float(np.cos(a)) * sp,
                                float(np.sin(a)) * sp, life, life,
                                0.9 + 1.8 * rng.random(), self._cur_hue])

    def _update_fx(self, dt):
        if self._intro > 0:
            self._intro = max(0.0, self._intro - 1.5 * dt)
        if self._particles:
            keep = []
            for q in self._particles:
                q[0] += q[2] * dt
                q[1] += q[3] * dt
                q[2] *= 0.9
                q[3] *= 0.9
                q[4] -= dt
                if q[4] > 0:
                    keep.append(q)
            self._particles = keep

    # ---------------- animation ----------------
    def _tick(self):
        self._elapsed += self._timer.interval()
        dt = self._timer.interval() / 1000.0
        self._update_fx(dt)
        m = self.model
        if m is not None and self.playing:
            dt_t = self.speed * dt / self.LOOP_SECONDS
            n_sub = int(np.clip(np.ceil(dt_t * m.n_total * 2.0), 1, 64))
            dts = dt_t / n_sub
            for _ in range(n_sub):
                if self.t >= 1.0 and not self.loop:
                    break
                self._advance(dts)
            self.timeChanged.emit(self.t)
        self.update()

    def _advance(self, dt_s):
        m = self.model
        nt = self.t + dt_s
        if nt >= 1.0:
            if self.loop:
                self._reset_artwork()
                self.t = 0.0
            else:
                self.t = 1.0
                self._rebuild_trail()
                self.set_playing(False)
            return

        kind, idx, t0, t1 = m.seg_at(self.t)
        if kind == "path":
            if self._active_idx != idx:
                self._begin_path(idx)
            path = m.paths[idx]
            u = (self.t - t0) / max(t1 - t0, 1e-9)
            self._phase = u
            z = path.evaluate_point(u, self.terms)
            self._pen = QPointF(z.real, z.imag)
            self._cur.append(self._pen)
            if self.sparkles and self._rng.random() < 0.55:
                self._spawn(z)
        else:
            if self._in_jump is None:
                self._end_path()
                a, b = m.path_start(idx), m.path_start(idx + 1)
                self._jump_lines.append((QPointF(a.real, a.imag),
                                         QPointF(b.real, b.imag)))
                self._in_jump = (a, b)
                self._phase = None
            a, b = self._in_jump
            v = (self.t - t0) / max(t1 - t0, 1e-9)
            self._pen = QPointF(a.real + (b.real - a.real) * v,
                                a.imag + (b.imag - a.imag) * v)

    def mousePressEvent(self, e):
        if self.model is None:
            return
        if self.playing:
            self.set_playing(False)
        else:
            if self.t >= 0.999 and not self.loop:
                self.set_time(0.0)
            self.set_playing(True)

    # ---------------- backdrop ----------------
    def _rebuild_backdrop(self):
        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            self._backdrop = None
            return
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        if self.paper:
            g = QLinearGradient(0, 0, 0, h)
            g.setColorAt(0.0, QColor("#f8f3e8"))
            g.setColorAt(1.0, QColor("#eae1cb"))
            p.fillRect(img.rect(), QBrush(g))
            vg = QRadialGradient(w / 2.0, h / 2.0, max(w, h) * 0.72)
            vg.setColorAt(0.55, QColor(60, 40, 20, 0))
            vg.setColorAt(1.0, QColor(60, 40, 20, 55))
            p.fillRect(img.rect(), QBrush(vg))
        else:
            g = QRadialGradient(w * 0.5, h * 0.45, max(w, h) * 0.75)
            g.setColorAt(0.0, QColor(19, 29, 50))
            g.setColorAt(1.0, QColor(6, 9, 17))
            p.fillRect(img.rect(), QBrush(g))
            vg = QRadialGradient(w / 2.0, h / 2.0, max(w, h) * 0.72)
            vg.setColorAt(0.55, QColor(0, 0, 0, 0))
            vg.setColorAt(1.0, QColor(0, 0, 0, 140))
            p.fillRect(img.rect(), QBrush(vg))
        p.end()
        self._backdrop = img

    def resizeEvent(self, e):
        self._rebuild_backdrop()
        self._layers_dirty = True
        super().resizeEvent(e)

    # ---------------- offscreen rendering ----------------
    def render_to_image(self, w, h):
        img = QImage(int(w), int(h), QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.black)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_frame(p, int(w), int(h))
        p.end()
        return img

    # ---------------- PNG export ----------------
    def export_png(self, path, paper):
        m = self.model
        if m is None:
            return False
        entries = list(self._done)
        if self._cur.size() >= 2:
            entries.append((self._cur, self._cur_hue))
        if not entries:
            return False
        S, sc = int(m.size), 4
        W = H = S * sc
        img = QImage(W, H, QImage.Format_ARGB32)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if paper:
            img.fill(QColor("#f6f0e3"))
        else:
            g = QRadialGradient(W / 2, H / 2, max(W, H) * 0.7)
            g.setColorAt(0.0, QColor(16, 25, 44))
            g.setColorAt(1.0, QColor(7, 10, 18))
            p.fillRect(img.rect(), QBrush(g))
        p.translate(W / 2, H / 2)
        p.scale(sc, sc)
        if paper:
            p.setPen(QPen(QColor(32, 30, 38, 240), 1.5))
            for poly, _ in entries:
                p.drawPolyline(poly)
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            for wd, al in ((10, 26), (4.5, 60)):
                for poly, hue in entries:
                    p.setPen(QPen(self._glow_col(hue, al), wd))
                    p.drawPolyline(poly)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            for poly, hue in entries:
                p.setPen(QPen(self._ink(hue, 240), 1.7))
                p.drawPolyline(poly)
        p.end()
        return img.save(path)

    # ---------------- painting ----------------
    def paintEvent(self, event):
        p = QPainter(self)
        self.paint_frame(p, self.width(), self.height())
        p.end()

    def paint_frame(self, p, w, h):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if (self._backdrop is None or self._backdrop.width() != w
                or self._backdrop.height() != h):
            self._rebuild_backdrop()
        if self._backdrop is not None:
            p.drawImage(0, 0, self._backdrop)

        if self.show_stars and not self.paper and self._stars.size:
            p.setPen(Qt.PenStyle.NoPen)
            el = self._elapsed * 0.001
            for x, y, r, a0, ph, spd in self._stars.tolist():
                a = int(a0 * (0.55 + 0.45 * math.sin(el * spd + ph)))
                if a <= 0:
                    continue
                p.setBrush(QColor(255, 255, 255, a))
                p.drawEllipse(QPointF(x * w, y * h), r, r)

        m = self.model
        if m is None:
            p.setPen(QColor("#8aa0b4"))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter,
                       "Load an image to begin.\n\n'Load image…' for a local file,\n"
                       "or use the one-click downloads in the panel.")
            return
        if self._layers_dirty:
            self._rebake_layers()

        lay = self._layout(w, h)
        rect = lay["stage"]
        S = m.size
        s = max(0.01, rect.width() / S)
        cx, cy = rect.center().x(), rect.center().y()
        k = self._intro
        ease = k * k * (3.0 - 2.0 * k)
        zoom = 1.0 + 0.10 * ease
        alpha = 1.0 - 0.5 * ease
        vs = s * zoom

        def cosmetic(color, width, dash=False):
            q = QPen(color)
            q.setCosmetic(True)
            q.setWidthF(width)
            if dash:
                q.setStyle(Qt.PenStyle.DashLine)
                q.setDashPattern([2.5, 3.5])
            return q

        p.save()
        p.setOpacity(alpha)

        # ghost / intro teaser
        ia = self.intro_card
        ghost_a = (int(120 * min(1.0, ia * 1.25)) if ia > 0
                   else (34 if self.show_ghost else 0))
        if ghost_a > 0:
            p.save()
            p.translate(cx, cy)
            p.scale(vs, vs)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(cosmetic(QColor(60, 50, 30, ghost_a) if self.paper
                              else QColor(255, 255, 255, ghost_a), 1.2,
                              dash=(ia <= 0)))
            for poly in m.ghost_polys:
                if poly.size() >= 2:
                    p.drawPolyline(poly)
            p.restore()

        if self.bloom and not self.paper and self._glow is not None:
            p.save()
            p.translate(cx, cy)
            p.scale(zoom, zoom)
            p.translate(-cx, -cy)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            p.drawImage(QRectF(0, 0, w, h), self._glow)
            p.restore()

        if self._core is not None:
            p.save()
            p.translate(cx, cy)
            p.scale(zoom, zoom)
            p.translate(-cx, -cy)
            p.setOpacity(alpha * 0.98)
            p.drawImage(0, 0, self._core)
            p.restore()

        p.save()
        p.translate(cx, cy)
        p.scale(vs, vs)
        additive = not self.paper
        hue = self._cur_hue

        if self.show_jumps and self._jump_lines:
            pen = cosmetic(QColor(90, 90, 110, 85) if self.paper
                           else QColor(150, 175, 225, 95), 1.1, dash=True)
            pen.setDashOffset(-(self._elapsed * 0.02))
            p.setPen(pen)
            for a, b in self._jump_lines:
                p.drawLine(a, b)

        if additive:
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        if self._cur.size() >= 2:
            passes = [(7.0, 26), (3.4, 70), (1.6, 242)] if additive \
                else [(1.5, 235)]
            for wd, al in passes:
                pen = cosmetic(self._ink(hue, al), wd)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawPolyline(self._cur)

        if ((self.show_circles or self.show_arms) and self._phase is not None
                and self._in_jump is None and self._active_idx >= 0):
            path = m.paths[self._active_idx]
            chain = path.tip_chain(self._phase, self.terms)
            if self.show_circles and chain.size:
                ccol = (hue_color(hue, 0.6, 0.5, 65) if (hue is not None and self.paper)
                        else hue_color(hue, 0.55, 1.0, 70) if hue is not None
                        else QColor(30, 80, 190, 45) if self.paper
                        else QColor(110, 150, 255, 60))
                p.setPen(cosmetic(ccol, 1.0))
                min_r = 1.4 / vs
                for kk in range(chain.size):
                    rad = float(path.amp[kk])
                    if rad > min_r:
                        c = chain[kk]
                        p.drawEllipse(QPointF(float(c.real), float(c.imag)),
                                      rad, rad)
            if self.show_arms and chain.size:
                aa = 150 if self.terms <= 80 else max(55, 150 - (self.terms - 80) // 2)
                acol = (hue_color(hue, 0.6, 0.45, aa * 2 // 3) if (hue is not None and self.paper)
                        else hue_color(hue, 0.45, 1.0, aa) if hue is not None
                        else QColor(30, 80, 190, aa * 2 // 3) if self.paper
                        else QColor(160, 195, 255, aa))
                p.setPen(cosmetic(acol, 1.1))
                prev = 0j
                for kk in range(chain.size):
                    c = complex(chain[kk])
                    if abs(c - prev) * vs > 0.8:
                        p.drawLine(QPointF(prev.real, prev.imag),
                                   QPointF(c.real, c.imag))
                    prev = c

        if self.sparkles and self._particles and additive:
            p.setPen(Qt.PenStyle.NoPen)
            for q in self._particles:
                lf = q[4] / q[5]
                col = hue_color(q[7], 0.35, 1.0, int(190 * lf)) \
                    if q[7] is not None else QColor(255, 240, 200, int(190 * lf))
                p.setBrush(col)
                rr = q[6] * (0.35 + 0.65 * lf)
                p.drawEllipse(QPointF(q[0], q[1]), rr, rr)

        if self._pen is not None:
            pulse = 1.0 + 0.16 * math.sin(self._elapsed * 0.006)
            pr = 4.2 * pulse / vs
            if additive:
                rg = QRadialGradient(self._pen, pr * 2.6)
                core = QColor(255, 255, 255, 235)
                mid = hue_color(hue, 0.55, 1.0, 200) if hue is not None \
                    else QColor(255, 190, 110, 200)
                rg.setColorAt(0.0, core)
                rg.setColorAt(0.35, mid)
                rg.setColorAt(1.0, QColor(mid.red(), mid.green(), mid.blue(), 0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(rg))
                p.drawEllipse(self._pen, pr * 2.6, pr * 2.6)
                p.setBrush(QColor(255, 255, 255, 220))
                p.drawEllipse(self._pen, pr * 0.4, pr * 0.4)
            else:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(200, 30, 20) if not self.rainbow
                           else hue_color(hue, 0.7, 0.4, 255))
                p.drawEllipse(self._pen, pr * 0.8, pr * 0.8)

        p.restore()
        p.restore()

        # ---- HUD (live canvas only) ----
        if self.show_hud and not self.video_mode:
            ring = QRectF(9.5, 9.5, w - 19, h - 19)
            p.setPen(QPen(QColor(60, 50, 30, 50) if self.paper
                          else QColor(255, 255, 255, 26), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(ring)
            if self.t > 0.0005:
                span = -int(self.t * 360 * 16)
                acc = QColor(48, 80, 192) if self.paper else QColor(138, 180, 248)
                p.setPen(QPen(QColor(acc.red(), acc.green(), acc.blue(), 70), 4.0))
                p.drawArc(ring, 90 * 16, span)
                p.setPen(QPen(acc, 2.2))
                p.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.drawArc(ring, 90 * 16, span)
            n_paths = len(m.paths)
            cur = self._active_idx + 1 if self._active_idx >= 0 else 0
            txt = (f"{cur}/{n_paths} paths   ·   {self.terms:,} epicycles   "
                   f"·   {self.speed:.2f}×")
            if not self.playing:
                txt += "   ·   paused"
            fm = QFontMetrics(p.font())
            tw = fm.horizontalAdvance(txt)
            pad, hgt = 12, fm.height() + 12
            pill = QRectF(16, h - 16 - hgt, tw + 2 * pad, hgt)
            p.setPen(QColor(58, 83, 120, 120))
            p.setBrush(QColor(255, 255, 255, 185) if self.paper
                       else QColor(8, 12, 20, 180))
            p.drawRoundedRect(pill, 8, 8)
            p.setPen(QColor(74, 66, 52) if self.paper else QColor(215, 228, 246))
            p.drawText(pill, Qt.AlignmentFlag.AlignCenter, txt)

        # ---- YouTube/Shorts overlays ----
        if self.show_video_ui and self.intro_card <= 0:
            self._draw_video_ui(p, w, h, lay)
        if self.intro_card > 0:
            self._draw_intro(p, w, h, lay)
        if self.outro_card > 0:
            self._draw_outro(p, w, h, lay)

    # ---------------- text helper ----------------
    def _font(self, px, bold=False, spacing=0.0):
        f = QFont()
        f.setFamilies(["Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue",
                       "Arial"])
        f.setPixelSize(max(9, int(px)))
        f.setBold(bold)
        if spacing:
            f.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
        return f

    def _draw_text(self, p, txt, rect_y, w, px, color, bold=False, ls=0.0,
                   align=Qt.AlignmentFlag.AlignHCenter, shadow=True):
        f = self._font(px, bold, ls)
        p.setFont(f)
        fm = QFontMetrics(f)
        fh = fm.height()
        r = QRectF(0, rect_y, w, fh * 1.35)
        if shadow:
            p.setPen(QColor(0, 0, 0, 160))
            p.drawText(r.translated(0, max(1.5, px * 0.055)), align, txt)
        p.setPen(color)
        p.drawText(r, align, txt)
        return fh

    # ---------------- video overlays ----------------
    def _draw_video_ui(self, p, w, h, lay):
        m = self.model
        acc = QColor(48, 80, 192) if self.paper else QColor(138, 180, 248)
        cap_col = QColor(70, 62, 48, 235) if self.paper else QColor(214, 228, 246, 235)

        # watermark (top-right, above Shorts' side buttons zone)
        if self.video_handle:
            self._draw_text(p, self.video_handle, 0.030 * h, w - 14,
                            max(12, h * 0.019),
                            QColor(90, 84, 66, 160) if self.paper
                            else QColor(255, 255, 255, 120),
                            align=Qt.AlignmentFlag.AlignRight, shadow=False)

        bar_y = h - max(22.0, 0.058 * h)
        cap_y = bar_y - 0.048 * h

        if self.outro_card <= 0:
            # caption
            if self.t < 0.999:
                P = len(m.paths)
                cur = max(1, self._active_idx + 1)
                cap = f"stroke {cur:,} / {P:,}   ·   {self.terms:,} epicycles"
            else:
                cap = (f"{len(m.paths):,} strokes   ·   {self.terms:,} "
                       f"epicycles each")
            self._draw_text(p, cap, cap_y, w, max(13, h * 0.021), cap_col,
                            bold=False, ls=0.4)

            # progress bar
            bw = min(w * 0.52, 640.0)
            bh = max(5.0, h * 0.0045)
            x = (w - bw) / 2.0
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(120, 110, 90, 60) if self.paper
                       else QColor(255, 255, 255, 30))
            p.drawRoundedRect(QRectF(x, bar_y, bw, bh), bh / 2, bh / 2)
            if self.t > 0:
                p.setBrush(acc)
                p.drawRoundedRect(QRectF(x, bar_y, max(bh, bw * self.t), bh),
                                  bh / 2, bh / 2)

    def _draw_intro(self, p, w, h, lay):
        k = self.intro_card
        a_in = min(1.0, k / 0.20)
        a_out = min(1.0, (1.0 - k) / 0.20)
        a = a_in * a_out
        if a <= 0.001:
            return
        p.setOpacity(a)

        if self.video_title:
            self._draw_text(p, self.video_title, lay["title_y"], w,
                            max(20, h * 0.043),
                            QColor(255, 255, 255) if not self.paper
                            else QColor(38, 32, 24),
                            bold=True, ls=1.6)
        if self.video_sub:
            self._draw_text(p, self.video_sub, lay["sub_y"], w,
                            max(13, h * 0.021),
                            QColor(178, 200, 232) if not self.paper
                            else QColor(96, 84, 62),
                            ls=0.8)

        # small accent rule under subtitle
        rw = min(w * 0.16, 190.0)
        ry = lay["sub_y"] + max(30, h * 0.030)
        g = QLinearGradient((w - rw) / 2, 0, (w + rw) / 2, 0)
        g.setColorAt(0.0, QColor(138, 180, 248, 0))
        g.setColorAt(0.5, QColor(138, 180, 248, 220))
        g.setColorAt(1.0, QColor(138, 180, 248, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        p.drawRect(QRectF((w - rw) / 2, ry, rw, 2.2))

        if self.video_handle:
            self._draw_text(p, self.video_handle, 0.030 * h, w - 14,
                            max(12, h * 0.019),
                            QColor(255, 255, 255, 150) if not self.paper
                            else QColor(90, 84, 66, 160),
                            align=Qt.AlignmentFlag.AlignRight, shadow=False)
        p.setOpacity(1.0)

    def _draw_outro(self, p, w, h, lay):
        k = self.outro_card
        e = k * k * (3.0 - 2.0 * k)
        p.setOpacity(min(1.0, k * 1.5))

        acc = QColor(48, 80, 192) if self.paper else QColor(138, 180, 248)
        bar_y = h - max(22.0, 0.058 * h)

        # Subscribe pill
        pw = min(w * 0.42, 460.0)
        ph = max(54.0, h * 0.042)
        px = (w - pw) / 2.0
        py = bar_y - 0.115 * h + (1.0 - e) * h * 0.045
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 0, 51, int(238 * e)))
        p.drawRoundedRect(QRectF(px, py, pw, ph), ph / 2, ph / 2)
        self._draw_text(p, "SUBSCRIBE", py + ph * 0.26, w, max(15, h * 0.023),
                        QColor(255, 255, 255, int(255 * e)), bold=True, ls=3.0)

        self._draw_text(p, "like & follow for more Fourier art",
                        py + ph + 0.018 * h, w, max(12, h * 0.018),
                        QColor(200, 216, 240, int(210 * e)) if not self.paper
                        else QColor(96, 84, 62, int(210 * e)), ls=0.6)

        # gentle accent pulse under the finished drawing
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(acc.red(), acc.green(), acc.blue(), int(26 * e)))
        pr = lay["stage"].width() * (0.52 + 0.02 * math.sin(k * 6.0))
        rg = QRadialGradient(w / 2.0, lay["stage"].center().y(), pr)
        rg.setColorAt(0.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(rg))
        p.drawEllipse(QPointF(w / 2.0, lay["stage"].center().y()), pr, pr)
        p.setOpacity(1.0)


# ===========================================================================
# Video writers / backend detection
# ===========================================================================
def _qimage_to_rgb(img):
    img = img.convertToFormat(QImage.Format_RGB888)
    h, w, bpl = img.height(), img.width(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), np.uint8, h * bpl).reshape(h, bpl)
    return np.ascontiguousarray(arr[:, :w * 3].reshape(h, w, 3))


def video_backends():
    out = []
    if _HAS_IMAGEIO:
        out.append(("mp4_imageio", "MP4 · imageio-ffmpeg"))
    if shutil.which("ffmpeg"):
        out.append(("mp4_ffmpeg", "MP4 · system ffmpeg"))
    if _HAS_PIL:
        out.append(("gif", "Animated GIF · Pillow"))
    out.append(("pngseq", "PNG frame sequence (folder)"))
    return out


_QUAL_CRF = {"high": 16, "balanced": 18, "small": 22}
_QUAL_IMGIO = {"high": 9, "balanced": 7, "small": 5}


class ImageioWriter:
    def __init__(self, path, fps, quality="balanced"):
        self.err = ""
        q = _QUAL_IMGIO.get(quality, 7)
        try:
            self.w = imageio.get_writer(path, mode="I", fps=fps, codec="libx264",
                                        quality=q, pixelformat="yuv420p",
                                        macro_block_size=1)
        except TypeError:
            self.w = imageio.get_writer(path, mode="I", fps=fps)

    def write(self, arr):
        self.w.append_data(arr)

    def close(self):
        try:
            self.w.close()
            return True
        except Exception as exc:
            self.err = str(exc)
            return False


class FfmpegPipeWriter:
    def __init__(self, path, fps, w, h, quality="balanced"):
        self.err = ""
        self.failed = False
        self._tmp = tempfile.TemporaryFile()
        crf = _QUAL_CRF.get(quality, 18)
        cmd = [shutil.which("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "-", "-an",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
               "-profile:v", "high", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._tmp)

    def write(self, arr):
        if self.failed:
            return
        try:
            self.proc.stdin.write(arr.tobytes())
        except (BrokenPipeError, OSError):
            self.failed = True

    def close(self):
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass
        rc = self.proc.wait()
        if rc == 0:
            return True
        self._tmp.seek(0)
        tail = self._tmp.read()[-800:].decode("utf8", "replace")
        self._tmp.close()
        self.err = f"ffmpeg exited with {rc}: {tail}"
        return False


class GifWriter:
    def __init__(self, path, fps, quality="balanced"):
        self.path = path
        self.delay = max(20, int(round(1000.0 / fps)))
        self.frames = []
        self.master = None
        self.err = ""

    def write(self, arr):
        im = Image.fromarray(arr).convert("RGB")
        if self.master is None:
            self.master = im.quantize(colors=256)
            self.frames.append(self.master)
        else:
            self.frames.append(im.quantize(palette=self.master,
                                           dither=_GIF_DITHER))

    def close(self):
        if not self.frames:
            self.err = "no frames"
            return False
        try:
            self.frames[0].save(self.path, save_all=True,
                                append_images=self.frames[1:],
                                duration=self.delay, loop=0)
            return True
        except Exception as exc:
            self.err = str(exc)
            return False


class PngSeqWriter:
    def __init__(self, folder, fps, w, h, quality="balanced"):
        self.folder = folder
        self.w, self.h = w, h
        self.n = 0
        self.err = ""

    def write(self, arr):
        self.n += 1
        data = arr.tobytes()
        img = QImage(data, self.w, self.h, 3 * self.w, QImage.Format_RGB888)
        img.save(os.path.join(self.folder, f"frame_{self.n:05d}.png"))

    def close(self):
        return self.n > 0


def make_writer(fmt, path, fps, w, h, quality="balanced"):
    if fmt == "mp4_imageio":
        return ImageioWriter(path, fps, quality)
    if fmt == "mp4_ffmpeg":
        return FfmpegPipeWriter(path, fps, w, h, quality)
    if fmt == "gif":
        return GifWriter(path, fps, quality)
    if fmt == "pngseq":
        return PngSeqWriter(path, fps, w, h, quality)
    raise ValueError(f"Unknown format: {fmt}")


# ===========================================================================
# Export canvas + worker
# ===========================================================================
def make_export_canvas(model, terms, w, h, opts):
    c = FourierCanvas()
    c._timer.stop()
    c.resize(int(w), int(h))
    c.model = model
    c.terms = int(terms)
    c.loop = False
    c.playing = True
    c.speed = 1.0
    c._intro = 0.0
    c.video_mode = True
    for key, val in opts.items():
        setattr(c, key, val)
    c._rebuild_backdrop()
    c._reset_artwork()
    c._rebake_layers()
    c.t = 0.0
    return c


class VideoExportJob(QObject):
    progress = Signal(int, int, str)        # done, total, eta string
    finished = Signal(bool, str)

    def __init__(self, shadow, model, settings):
        super().__init__()
        self.shadow = shadow
        self.model = model
        self.s = settings
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        ok, msg = False, ""
        try:
            s = self.s
            w, h, fps = s["w"], s["h"], s["fps"]
            fi = max(0, int(round(s["intro_s"] * fps)))
            fd = max(1, int(round(s["draw_s"] * fps)))
            fh = max(0, int(round(s["hold_s"] * fps)))
            fo = max(0, int(round(s["outro_s"] * fps)))
            total = fi + fd + fh + fo
            writer = make_writer(s["fmt"], s["path"], fps, w, h, s["quality"])
            shadow, model = self.shadow, self.model
            n_total = max(1, model.n_total)
            t0 = time.perf_counter()

            for i in range(total):
                if self._cancel:
                    break
                if i < fi:                                   # intro card
                    shadow.intro_card = (i + 1) / fi
                    shadow.outro_card = 0.0
                    if shadow.t != 0.0:
                        shadow.set_time(0.0)
                elif i < fi + fd:                            # drawing
                    shadow.intro_card = 0.0
                    shadow.outro_card = 0.0
                    target = (i - fi) / (fd - 1) if fd > 1 else 0.0
                    if shadow.t < target - 1e-9:
                        step = target - shadow.t
                        n = int(np.clip(np.ceil(step * n_total * 2.5), 1, 400))
                        dts = step / n
                        for _ in range(n):
                            shadow._advance(dts)
                            if shadow.t >= 1.0:
                                break
                elif i < fi + fd + fh:                       # hold finished art
                    shadow.intro_card = 0.0
                    shadow.outro_card = 0.0
                    if shadow.t < 1.0:
                        shadow.set_time(1.0)
                else:                                        # outro card
                    shadow.intro_card = 0.0
                    shadow.outro_card = (i - (fi + fd + fh) + 1) / fo

                shadow._elapsed = int(round(i * 1000.0 / fps))
                shadow._update_fx(1.0 / fps)

                if s.get("ssaa") and w * h <= 1920 * 1080 * 2:
                    big = shadow.render_to_image(w * 2, h * 2)
                    frame = big.scaled(w, h,
                                       Qt.AspectRatioMode.IgnoreAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
                else:
                    frame = shadow.render_to_image(w, h)
                writer.write(_qimage_to_rgb(frame))

                done = i + 1
                dt = time.perf_counter() - t0
                eta = dt / done * (total - done)
                eta_s = f"ETA {int(eta // 60)}:{int(eta % 60):02d}"
                self.progress.emit(done, total, eta_s)

            done = writer.close()
            if self._cancel:
                ok = done
                msg = "Cancelled — partial file kept." if done else "Cancelled."
            elif done:
                ok = True
                msg = s["path"]
            else:
                msg = getattr(writer, "err", "") or "Writer failed."
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
        self.finished.emit(ok, msg)


# ===========================================================================
# Export dialog
# ===========================================================================
PLATFORMS = [
    ("YouTube Shorts  9:16", "1080x1920", "shorts"),
    ("YouTube HD  16:9", "1920x1080", "yt"),
    ("YouTube 720p  16:9", "1280x720", "yt720"),
    ("YouTube 1440p  16:9", "2560x1440", "yt1440"),
    ("YouTube 4K  16:9", "3840x2160", "yt4k"),
    ("Square  1:1", "1080x1080", "square"),
    ("Custom…", "", "custom"),
]
FPS_OPTIONS = [24, 30, 60]
QUALITIES = [("High bitrate", "high"), ("Balanced", "balanced"),
             ("Smaller file", "small")]


class VideoExportDialog(QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent or main)
        self.main = main
        self.setWindowTitle("Export video — YouTube / Shorts")
        self.setMinimumWidth(470)
        self._job = None
        self._thread = None
        self._running = False
        self._preview_canvas = None

        self._settings = QSettings("FourierEpicycles", "export")

        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        # ---- platform / format ----
        g1 = QGroupBox("Platform & format")
        f1 = QGridLayout(g1)
        f1.setVerticalSpacing(6)

        self.plat_combo = QComboBox()
        for name, res, tag in PLATFORMS:
            self.plat_combo.addItem(name, (res, tag))
        self.plat_combo.setCurrentIndex(0)                 # Shorts default
        self.plat_combo.currentIndexChanged.connect(self._platform_changed)

        self.fmt_combo = QComboBox()
        backends = video_backends()
        for key, label in backends:
            self.fmt_combo.addItem(label, key)
        pref = next((i for i, (k, _) in enumerate(backends)
                     if k.startswith("mp4")), 0)
        self.fmt_combo.setCurrentIndex(pref)

        self.fps_combo = QComboBox()
        for f in FPS_OPTIONS:
            self.fps_combo.addItem(f"{f} fps")
        self.fps_combo.setCurrentIndex(1)                  # 30

        self.qual_combo = QComboBox()
        for label, key in QUALITIES:
            self.qual_combo.addItem(label, key)
        self.qual_combo.setCurrentIndex(1)

        self.w_spin = QSpinBox()
        self.h_spin = QSpinBox()
        for sp in (self.w_spin, self.h_spin):
            sp.setRange(64, 4096)
            sp.setSingleStep(2)
        self.w_spin.setValue(1080)
        self.h_spin.setValue(1920)

        self.cust_w_lbl = QLabel("W")
        self.cust_h_lbl = QLabel("H")
        self._custom_widgets = (self.cust_w_lbl, self.w_spin,
                                self.cust_h_lbl, self.h_spin)

        f1.addWidget(QLabel("Platform"), 0, 0)
        f1.addWidget(self.plat_combo, 0, 1, 1, 3)
        f1.addWidget(QLabel("Format"), 1, 0)
        f1.addWidget(self.fmt_combo, 1, 1)
        f1.addWidget(QLabel("Frame rate"), 1, 2)
        f1.addWidget(self.fps_combo, 1, 3)
        f1.addWidget(QLabel("Quality"), 2, 0)
        f1.addWidget(self.qual_combo, 2, 1)
        f1.addWidget(self.cust_w_lbl, 2, 2)
        f1.addWidget(self.w_spin, 2, 2)
        f1.addWidget(self.cust_h_lbl, 2, 3)
        f1.addWidget(self.h_spin, 2, 3)
        lay.addWidget(g1)

        # ---- timing ----
        g2 = QGroupBox("Timeline")
        f2 = QGridLayout(g2)
        f2.setVerticalSpacing(6)
        self.intro_spin = self._sec_spin(0, 8, 2.0)
        self.draw_spin = self._sec_spin(3, 120, 14.0)
        self.hold_spin = self._sec_spin(0, 10, 2.5)
        self.outro_spin = self._sec_spin(0, 8, 2.5)
        for col, (lbl, sp) in enumerate((
                ("Intro", self.intro_spin), ("Draw", self.draw_spin),
                ("Hold", self.hold_spin), ("Outro", self.outro_spin))):
            f2.addWidget(QLabel(lbl), 0, col)
            f2.addWidget(sp, 1, col)
        lay.addWidget(g2)

        # ---- text ----
        g3 = QGroupBox("Title card & watermark")
        f3 = QGridLayout(g3)
        f3.setVerticalSpacing(6)
        self.title_edit = QLineEdit(self.main.source_name or "Fourier art")
        self.sub_edit = QLineEdit(self._settings.value(
            "subtitle", "drawn with Fourier epicycles"))
        self.handle_edit = QLineEdit(self._settings.value("handle", "@yourchannel"))
        f3.addWidget(QLabel("Title"), 0, 0)
        f3.addWidget(self.title_edit, 0, 1)
        f3.addWidget(QLabel("Subtitle"), 1, 0)
        f3.addWidget(self.sub_edit, 1, 1)
        f3.addWidget(QLabel("@handle"), 2, 0)
        f3.addWidget(self.handle_edit, 2, 1)
        lay.addWidget(g3)

        # ---- layers ----
        g4 = QGroupBox("Include in video")
        grid = QGridLayout(g4)
        grid.setVerticalSpacing(4)
        d = self.main.canvas
        self.cb_ui = QCheckBox("Captions + progress bar")
        self.cb_ui.setChecked(True)
        self.cb_epi = QCheckBox("Epicycles")
        self.cb_epi.setChecked(d.show_circles)
        self.cb_ghost = QCheckBox("Ghost paths")
        self.cb_glow = QCheckBox("Neon glow")
        self.cb_glow.setChecked(d.bloom)
        self.cb_stars = QCheckBox("Starfield")
        self.cb_stars.setChecked(d.show_stars)
        self.cb_spark = QCheckBox("Sparkles")
        self.cb_spark.setChecked(d.sparkles)
        self.cb_rain = QCheckBox("Rainbow ink")
        self.cb_rain.setChecked(d.rainbow)
        self.cb_paper = QCheckBox("Paper mode")
        self.cb_ssaa = QCheckBox("Supersample 2× (slow, crisper)")
        for i, cb in enumerate((self.cb_ui, self.cb_epi, self.cb_ghost,
                                self.cb_glow, self.cb_stars, self.cb_spark,
                                self.cb_rain, self.cb_paper, self.cb_ssaa)):
            grid.addWidget(cb, i // 3, i % 3)
        lay.addWidget(g4)

        # ---- preview ----
        prow = QHBoxLayout()
        self.preview_label = QLabel()
        self.preview_label.setObjectName("preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFixedHeight(250)
        self.preview_label.setMinimumWidth(160)
        prow.addWidget(self.preview_label, 1)
        self.preview_combo = QComboBox()
        self.preview_combo.addItems(["Intro title", "Mid drawing", "Outro card"])
        self.preview_combo.setCurrentIndex(1)
        pvrow = QHBoxLayout()
        pvrow.addWidget(QLabel("Preview:"))
        pvrow.addWidget(self.preview_combo, 1)
        lay.addLayout(pvrow)
        lay.addWidget(self.preview_label)

        self.est_label = QLabel("")
        self.est_label.setObjectName("val")
        lay.addWidget(self.est_label)
        self.eta_label = QLabel("")
        self.eta_label.setObjectName("status")
        lay.addWidget(self.eta_label)

        note = QLabel("Shorts layout keeps art clear of top/bottom UI overlays; "
                      "captions sit above the description zone. MP4 uploads "
                      "directly to YouTube/Shorts.")
        note.setObjectName("info")
        note.setWordWrap(True)
        lay.addWidget(note)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        lay.addWidget(self.progress)

        brow = QHBoxLayout()
        self.start_btn = QPushButton("Start export")
        self.start_btn.clicked.connect(self._start_or_cancel)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        brow.addWidget(self.start_btn)
        brow.addWidget(close_btn)
        lay.addLayout(brow)

        # wiring
        for sp in (self.intro_spin, self.draw_spin, self.hold_spin,
                   self.outro_spin):
            sp.valueChanged.connect(self._update_est)
            sp.valueChanged.connect(self._refresh_preview_later)
        for wdg in (self.plat_combo, self.fmt_combo, self.fps_combo,
                    self.qual_combo, self.w_spin, self.h_spin):
            if isinstance(wdg, QComboBox):
                wdg.currentIndexChanged.connect(self._update_est)
                wdg.currentIndexChanged.connect(self._refresh_preview_later)
            else:
                wdg.valueChanged.connect(self._update_est)
                wdg.valueChanged.connect(self._refresh_preview_later)
        for cb in (self.cb_ui, self.cb_epi, self.cb_ghost, self.cb_glow,
                   self.cb_stars, self.cb_spark, self.cb_rain, self.cb_paper):
            cb.toggled.connect(self._refresh_preview_later)
        for ed in (self.title_edit, self.sub_edit, self.handle_edit):
            ed.textChanged.connect(self._refresh_preview_later)
        self.preview_combo.currentIndexChanged.connect(self._refresh_preview_later)

        self._platform_changed()
        self._update_est()

        self._pv_timer = QTimer(self)
        self._pv_timer.setSingleShot(True)
        self._pv_timer.setInterval(220)
        self._pv_timer.timeout.connect(self._refresh_preview)

        self._refresh_preview_later()

    # ---------- helpers ----------
    def _sec_spin(self, lo, hi, val):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(0.5)
        sp.setValue(val)
        sp.setSuffix(" s")
        sp.setDecimals(1)
        return sp

    def _fmt(self):
        return self.fmt_combo.currentData()

    def _dims(self):
        res, tag = self.plat_combo.currentData()
        if tag == "custom":
            w, h = self.w_spin.value(), self.h_spin.value()
        else:
            w, h = (int(v) for v in res.split("x"))
        if self._fmt() in ("mp4_imageio", "mp4_ffmpeg"):
            w -= w % 2
            h -= h % 2
        return w, h

    def _tag(self):
        return self.plat_combo.currentData()[1]

    def _platform_changed(self):
        custom = self._tag() == "custom"
        for wdg in self._custom_widgets:
            wdg.setVisible(custom)
        self._update_est()
        self._refresh_preview_later()

    def _opts(self):
        return dict(
            show_video_ui=self.cb_ui.isChecked(),
            show_circles=self.cb_epi.isChecked(),
            show_arms=self.cb_epi.isChecked(),
            show_ghost=self.cb_ghost.isChecked(),
            bloom=self.cb_glow.isChecked(),
            show_stars=self.cb_stars.isChecked(),
            sparkles=self.cb_spark.isChecked(),
            rainbow=self.cb_rain.isChecked(),
            paper=self.cb_paper.isChecked(),
            show_hud=False,
        )

    def _update_est(self, *_):
        w, h = self._dims()
        fps = FPS_OPTIONS[self.fps_combo.currentIndex()]
        dur = (self.intro_spin.value() + self.draw_spin.value()
               + self.hold_spin.value() + self.outro_spin.value())
        total = int(round(dur * fps))
        size_txt = ""
        if self._fmt() in ("mp4_imageio", "mp4_ffmpeg"):
            mbps = (h / 1080.0) * (fps / 30.0) * \
                {"high": 16.0, "balanced": 11.0, "small": 6.0}[
                    self.qual_combo.currentData()]
            size_txt = f"  ·  ~{dur * mbps / 8:.0f} MB"
        self.est_label.setText(f"{total:,} frames · {dur:.1f}s · {fps} fps · "
                               f"{w}×{h}{size_txt}")
        self.progress.setMaximum(max(1, total))

    # ---------- live preview ----------
    def _refresh_preview_later(self, *_):
        self._pv_timer.start()

    def _refresh_preview(self):
        if self._running:
            return
        w, h = self._dims()
        scale = 300.0 / max(w, h)
        pw, ph = max(2, int(w * scale)), max(2, int(h * scale))
        opts = self._opts()
        opts["video_title"] = self.title_edit.text()
        opts["video_sub"] = self.sub_edit.text()
        opts["video_handle"] = self.handle_edit.text()
        c = make_export_canvas(self.main.canvas.model, self.main.canvas.terms,
                               pw, ph, opts)
        mode = self.preview_combo.currentIndex()
        if mode == 0:
            c.intro_card = 1.0
        elif mode == 2:
            c.set_time(1.0)
            c.outro_card = 1.0
        else:
            c.set_time(0.35)
        img = c.render_to_image(pw, ph)
        self.preview_label.setPixmap(QPixmap.fromImage(img))
        self._preview_canvas = c                     # keep alive

    # ---------- export ----------
    def _set_ui_enabled(self, on):
        for wdg in (self.plat_combo, self.fmt_combo, self.fps_combo,
                    self.qual_combo, self.w_spin, self.h_spin,
                    self.intro_spin, self.draw_spin, self.hold_spin,
                    self.outro_spin, self.title_edit, self.sub_edit,
                    self.handle_edit, self.cb_ui, self.cb_epi, self.cb_ghost,
                    self.cb_glow, self.cb_stars, self.cb_spark, self.cb_rain,
                    self.cb_paper, self.cb_ssaa, self.preview_combo):
            wdg.setEnabled(on)

    def _start_or_cancel(self):
        if self._running:
            if self._job is not None:
                self._job.cancel()
                self.start_btn.setText("Cancelling…")
                self.start_btn.setEnabled(False)
            return

        fmt = self._fmt()
        w, h = self._dims()
        fps = FPS_OPTIONS[self.fps_combo.currentIndex()]
        tag = self._tag()

        if fmt == "pngseq":
            path = QFileDialog.getExistingDirectory(
                self, "Choose output folder for PNG frames")
            if not path:
                return
        else:
            ext = ".gif" if fmt == "gif" else ".mp4"
            base = os.path.splitext(self.main.source_name)[0] or "fourier"
            if base == "demo portrait":
                base = "demo"
            suggested = f"fourier_{base}_{tag}_{w}x{h}{ext}"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save video", suggested,
                f"{'GIF image' if fmt == 'gif' else 'MP4 video'} (*{ext})")
            if not path:
                return
            if not path.lower().endswith(ext):
                path += ext

        self._settings.setValue("subtitle", self.sub_edit.text())
        self._settings.setValue("handle", self.handle_edit.text())

        opts = self._opts()
        opts["video_title"] = self.title_edit.text()
        opts["video_sub"] = self.sub_edit.text()
        opts["video_handle"] = self.handle_edit.text()

        shadow = make_export_canvas(self.main.canvas.model,
                                    self.main.canvas.terms, w, h, opts)
        settings = dict(fmt=fmt, path=path, w=w, h=h, fps=fps,
                        quality=self.qual_combo.currentData(),
                        intro_s=float(self.intro_spin.value()),
                        draw_s=float(self.draw_spin.value()),
                        hold_s=float(self.hold_spin.value()),
                        outro_s=float(self.outro_spin.value()),
                        ssaa=self.cb_ssaa.isChecked())

        self._job = VideoExportJob(shadow, self.main.canvas.model, settings)
        self._job.progress.connect(self._on_progress)
        self._job.finished.connect(self._on_done)
        self._set_ui_enabled(False)
        self.preview_label.setText("exporting…")
        self.preview_label.setPixmap(QPixmap())
        self.start_btn.setText("Cancel export")
        self.eta_label.setText("starting…")
        self._running = True
        self._thread = threading.Thread(target=self._job.run, daemon=True)
        self._thread.start()

    def _on_progress(self, done, total, eta):
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        pct = int(100 * done / max(1, total))
        self.eta_label.setText(f"{pct}%  ·  {eta}  ·  frame {done:,}/{total:,}")

    def _on_done(self, ok, msg):
        self._running = False
        self._set_ui_enabled(True)
        self.start_btn.setText("Start export")
        self.start_btn.setEnabled(True)
        if ok:
            self.progress.setValue(self.progress.maximum())
            self.eta_label.setText("done ✓")
            self.main._set_status(f"Video saved: {msg}")
            self._refresh_preview_later()
        else:
            self.main._set_status(f"Video export: {msg}")
            self._refresh_preview_later()

    def closeEvent(self, event):
        if self._running:
            event.ignore()
            self.main._set_status("Video export running — cancel it first.")


# ===========================================================================
# Main window
# ===========================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Transform Drawing — Epicycles (PySide6)")
        self.resize(1300, 860)

        self.source_image = None
        self.source_name = "image"
        self.user_loaded = False
        self._auto_fetch = False
        self._processing = False
        self._fetcher = None
        self._fetch_preset_name = ""
        self._slider_value_labels = {}

        self.canvas = FourierCanvas()
        self.canvas.timeChanged.connect(self._on_time)
        self.canvas.playingChanged.connect(self._on_playing)
        self.canvas.activePathChanged.connect(self._on_active)

        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(350)
        self._detail_timer.timeout.connect(self.reprocess)
        self._terms_timer = QTimer(self)
        self._terms_timer.setSingleShot(True)
        self._terms_timer.setInterval(140)
        self._terms_timer.timeout.connect(self._apply_terms)

        panel = QWidget()
        panel.setFixedWidth(340)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(12, 12, 12, 12)
        pl.setSpacing(8)

        title = QLabel("Fourier epicycles")
        title.setObjectName("title")
        pl.addWidget(title)

        self.preview = QLabel()
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedHeight(150)
        pl.addWidget(self.preview)

        row = QHBoxLayout()
        self.load_btn = QPushButton("Load image…")
        self.load_btn.clicked.connect(self.load_file)
        row.addWidget(self.load_btn)
        demo_btn = QPushButton("Demo portrait")
        demo_btn.clicked.connect(self._load_demo)
        row.addWidget(demo_btn)
        pl.addLayout(row)

        row2 = QHBoxLayout()
        self.mona_btn = QPushButton("Mona Lisa")
        self.mona_btn.clicked.connect(lambda: self.fetch_preset("Mona Lisa"))
        self.starry_btn = QPushButton("The Starry Night")
        self.starry_btn.clicked.connect(
            lambda: self.fetch_preset("The Starry Night"))
        row2.addWidget(self.mona_btn)
        row2.addWidget(self.starry_btn)
        pl.addLayout(row2)

        g0 = QGroupBox("Source")
        g0l = QVBoxLayout(g0)
        g0l.setSpacing(6)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Edge outlines", "edges")
        self.mode_combo.addItem("Tone contours (topo)", "tone")
        self.mode_combo.setToolTip(
            "Edge outlines: trace boundaries of strong gradients.\n"
            "Tone contours: nested brightness iso-lines — draws shading too.")
        self.mode_combo.currentIndexChanged.connect(
            lambda _=0: self._detail_timer.start())
        self.mode_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        g0l.addWidget(self.mode_combo)
        pl.addWidget(g0)

        g1 = QGroupBox("Drawing")
        g1l = QVBoxLayout(g1)
        g1l.setSpacing(6)
        self.detail_slider = self._add_slider(
            g1l, "Path points (total)", 300, 8000, 100, 2600,
            lambda v: f"{v:,}", self._detail_timer.start)
        self.sens_slider = self._add_slider(
            g1l, "Edge sensitivity", 0, 100, 1, 45,
            lambda v: f"{v}", self._detail_timer.start)
        self.terms_slider = self._add_slider(
            g1l, "Epicycles (per path)", 1, 1, 1, 1,
            lambda v: f"{v:,}", self._terms_timer.start)
        self.speed_slider = self._add_slider(
            g1l, "Speed", 25, 400, 5, 100, lambda v: f"{v/100:.2f}×",
            lambda v: setattr(self.canvas, "speed", v / 100.0))
        pl.addWidget(g1)

        g2 = QGroupBox("Playback")
        g2l = QVBoxLayout(g2)
        g2l.setSpacing(6)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Time"))
        trow.addStretch(1)
        self.time_label = QLabel("t = 0.000  ·  0%")
        self.time_label.setObjectName("val")
        trow.addWidget(self.time_label)
        g2l.addLayout(trow)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.time_slider.sliderPressed.connect(self._scrub_start)
        self.time_slider.sliderMoved.connect(
            lambda v: self.canvas.set_time(v / 1000.0))
        self.time_slider.sliderReleased.connect(self._scrub_end)
        g2l.addWidget(self.time_slider)

        brow = QHBoxLayout()
        self.play_btn = QPushButton("Pause")
        self.play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play_btn.clicked.connect(self.toggle_play)
        restart_btn = QPushButton("Restart")
        restart_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        restart_btn.clicked.connect(self.restart)
        brow.addWidget(self.play_btn)
        brow.addWidget(restart_btn)
        g2l.addLayout(brow)

        self.cb_loop = QCheckBox("Loop (redraw forever)")
        self.cb_loop.setChecked(True)
        self.cb_loop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cb_loop.toggled.connect(lambda on: setattr(self.canvas, "loop", on))
        g2l.addWidget(self.cb_loop)
        pl.addWidget(g2)

        g3 = QGroupBox("Display")
        g3l = QVBoxLayout(g3)
        g3l.setSpacing(5)

        def add_cb(text, attr, checked=True, extra=None):
            cb = QCheckBox(text)
            cb.setChecked(checked)
            cb.setFocusPolicy(Qt.FocusPolicy.NoFocus)

            def on(state, a=attr, ex=extra):
                setattr(self.canvas, a, state)
                if ex:
                    ex()
                self.canvas.update()
            cb.toggled.connect(on)
            g3l.addWidget(cb)
            return cb

        self.cb_circ = add_cb("Show epicycle circles", "show_circles")
        self.cb_arms = add_cb("Show radius arms", "show_arms")
        self.cb_ghost = add_cb("Show ghost of target paths", "show_ghost")
        self.cb_jump = add_cb("Show pen jump lines", "show_jumps")
        self.cb_rain = add_cb("Rainbow ink", "rainbow", True,
                              lambda: setattr(self.canvas, "_layers_dirty", True))
        self.cb_bloom = add_cb("Bloom / glow", "bloom")
        self.cb_spark = add_cb("Sparkle particles", "sparkles")
        self.cb_stars = add_cb("Starfield", "show_stars")
        self.cb_hud = add_cb("HUD & progress ring", "show_hud")
        self.paper_cb = add_cb(
            "Paper mode (light canvas)", "paper", False,
            lambda: self.canvas._rebuild_backdrop())
        pl.addWidget(g3)

        self.spectrum = SpectrumWidget()
        pl.addWidget(self.spectrum)

        erow = QHBoxLayout()
        png_btn = QPushButton("PNG…")
        png_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        png_btn.clicked.connect(self.export_png)
        self.video_btn = QPushButton("Export video / Shorts…")
        self.video_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.video_btn.clicked.connect(self.export_video)
        erow.addWidget(png_btn)
        erow.addWidget(self.video_btn)
        pl.addLayout(erow)

        self.status_label = QLabel("Starting…")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        pl.addWidget(self.status_label)

        info = QLabel(
            "Each contour becomes a path of points; (x, y) is read as the complex "
            "number x+iy and re-expressed as a Fourier series — a chain of "
            "rotating vectors (epicycles). The tip draws one loop, then the pen "
            "lifts (dashed line) and jumps to the next.\n\n"
            "Keys: Space play/pause · R restart · ←/→ scrub · ↑/↓ terms · "
            "C circles · A arms · G ghost · J jumps · N rainbow · B glow · "
            "S stars · H HUD · P paper · L loop · E PNG · V video")
        info.setObjectName("info")
        info.setWordWrap(True)
        pl.addWidget(info)
        pl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(354)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(8, 8, 8, 8)
        h.setSpacing(8)
        h.addWidget(self.canvas, 1)
        h.addWidget(scroll)
        self.setCentralWidget(central)

        self._add_shortcuts()

        self.set_source(make_demo_image(), "demo portrait")
        self._auto_fetch = True
        QTimer.singleShot(300, lambda: self.fetch_preset("Mona Lisa"))

    # ---------------- helpers ----------------
    def _add_slider(self, layout, name, lo, hi, step, value, fmt, on_change=None):
        row = QHBoxLayout()
        row.addWidget(QLabel(name))
        row.addStretch(1)
        val = QLabel(fmt(value))
        val.setObjectName("val")
        row.addWidget(val)
        layout.addLayout(row)

        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setPageStep(step * 5)
        s.setValue(value)
        s.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider_value_labels[s] = val

        def changed(v):
            val.setText(fmt(v))
            if on_change:
                on_change()

        s.valueChanged.connect(changed)
        layout.addWidget(s)
        return s

    def _add_shortcuts(self):
        def sc(key, fn):
            q = QShortcut(QKeySequence(key), self)
            q.activated.connect(fn)

        sc(Qt.Key_Space, self.toggle_play)
        sc(Qt.Key_R, self.restart)
        sc(Qt.Key_Left, lambda: self.canvas.set_time(max(0.0, self.canvas.t - 0.01)))
        sc(Qt.Key_Right, lambda: self.canvas.set_time(min(1.0, self.canvas.t + 0.01)))
        sc(Qt.Key_Up, lambda: self._nudge_terms(+1))
        sc(Qt.Key_Down, lambda: self._nudge_terms(-1))
        sc(Qt.Key_C, self.cb_circ.toggle)
        sc(Qt.Key_A, self.cb_arms.toggle)
        sc(Qt.Key_G, self.cb_ghost.toggle)
        sc(Qt.Key_J, self.cb_jump.toggle)
        sc(Qt.Key_N, self.cb_rain.toggle)
        sc(Qt.Key_B, self.cb_bloom.toggle)
        sc(Qt.Key_S, self.cb_stars.toggle)
        sc(Qt.Key_H, self.cb_hud.toggle)
        sc(Qt.Key_P, self.paper_cb.toggle)
        sc(Qt.Key_L, self.cb_loop.toggle)
        sc(Qt.Key_E, self.export_png)
        sc(Qt.Key_V, self.export_video)

    def _nudge_terms(self, sign):
        s = self.terms_slider
        step = max(1, s.maximum() // 25)
        s.setValue(int(min(s.maximum(), max(s.minimum(), s.value() + sign * step))))

    def _set_status(self, text):
        self.status_label.setText(text)

    def _set_preset_buttons(self, enabled):
        self.mona_btn.setEnabled(enabled)
        self.starry_btn.setEnabled(enabled)

    # ---------------- sources ----------------
    def set_source(self, image, name=None):
        self.source_image = image
        self.source_name = name or "image"
        self.setWindowTitle(f"Fourier Drawing — {self.source_name}")
        pm = QPixmap.fromImage(image.scaled(
            318, 140, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self.preview.setPixmap(pm)
        self.reprocess()

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open an image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff);;"
            "All files (*)")
        if not path:
            return
        img = QImage(path)
        if img.isNull():
            QMessageBox.warning(self, "Fourier drawing", f"Could not load:\n{path}")
            return
        self.user_loaded = True
        self.set_source(img, os.path.basename(path))

    def _load_demo(self):
        self.user_loaded = True
        self.set_source(make_demo_image(), "demo portrait")

    def fetch_preset(self, name):
        if self._fetcher is not None and self._fetcher.running:
            return
        self._set_preset_buttons(False)
        self._set_status(f'Downloading "{name}" from Wikimedia Commons…')
        self._fetch_preset_name = name
        self._fetcher = ImageFetcher(PRESETS[name])
        self._fetcher.fetched.connect(self._on_fetched)
        self._fetcher.start()

    def _on_fetched(self, payload):
        auto, self._auto_fetch = self._auto_fetch, False
        self._set_preset_buttons(True)
        if isinstance(payload, str):
            if not auto:
                QMessageBox.warning(self, "Download failed", payload)
            self._set_status("Download failed (offline?). Use 'Load image…' "
                             "or the demo portrait.")
            return
        if auto and self.user_loaded:
            self._set_status("Image downloaded — kept the image you had loaded.")
            return
        self.set_source(payload, self._fetch_preset_name)

    # ---------------- processing ----------------
    def reprocess(self):
        if self.source_image is None or self._processing:
            return
        self._processing = True
        self._set_status("Tracing contours & computing Fourier series…")
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            model = build_model(
                self.source_image, size=480,
                detail=self.detail_slider.value(),
                sensitivity=self.sens_slider.value(),
                mode=self.mode_combo.currentData())
            err = None
        except Exception as exc:
            model, err = None, str(exc)
        finally:
            QApplication.restoreOverrideCursor()
            self._processing = False
        if model is None:
            self._set_status(f"Error: {err}")
            return

        maxn = max(p.n for p in model.paths)
        terms = min(500, maxn)
        self.terms_slider.blockSignals(True)
        self.terms_slider.setRange(1, max(1, maxn))
        self.terms_slider.setValue(terms)
        self.terms_slider.blockSignals(False)
        self._slider_value_labels[self.terms_slider].setText(f"{terms:,}")

        self.canvas.set_model(model)
        self.canvas.set_terms(terms)
        self._set_status(
            f"{self.source_name}: {len(model.paths)} paths · "
            f"{model.n_total:,} pts · terms {terms:,} / {maxn:,}")

    def _apply_terms(self):
        v = self.terms_slider.value()
        self.canvas.set_terms(v)
        self.spectrum.set_marker(v)

    def _on_active(self, idx):
        m = self.canvas.model
        if m is None or idx < 0 or idx >= len(m.paths):
            self.spectrum.set_data(None, 0)
        else:
            self.spectrum.set_data(m.paths[idx].amp, self.canvas.terms)

    # ---------------- playback ----------------
    def toggle_play(self):
        if self.canvas.model is None:
            return
        if self.canvas.playing:
            self.canvas.set_playing(False)
        else:
            if self.canvas.t >= 0.999 and not self.canvas.loop:
                self.canvas.set_time(0.0)
            self.canvas.set_playing(True)

    def restart(self):
        self.canvas.set_time(0.0)
        self.canvas.set_playing(True)

    def _scrub_start(self):
        self._was_playing = self.canvas.playing
        self.canvas.set_playing(False)

    def _scrub_end(self):
        self.canvas.set_playing(self._was_playing)

    def _on_time(self, t):
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(round(t * 1000)))
        self.time_slider.blockSignals(False)
        self.time_label.setText(f"t = {t:.3f}  ·  {int(round(t * 100))}%")

    def _on_playing(self, on):
        self.play_btn.setText("Pause" if on else "Play")

    # ---------------- export ----------------
    def export_png(self):
        if self.canvas.model is None:
            QMessageBox.information(self, "Export", "Nothing to export yet.")
            return
        base = os.path.splitext(self.source_name)[0] or "fourier"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save drawing", f"fourier_{base}.png", "PNG image (*.png)")
        if not path:
            return
        if self.canvas.export_png(path, self.paper_cb.isChecked()):
            self._set_status(f"Saved: {path}")
        else:
            QMessageBox.warning(self, "Export", "The drawing is empty — "
                                "let it play for a moment first.")

    def export_video(self):
        if self.canvas.model is None:
            QMessageBox.information(self, "Export", "Nothing to export yet.")
            return
        dlg = VideoExportDialog(self)
        dlg.exec()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
"""
face_detector.py -- face cropping for rPPG preprocessing.

Primary: MediaPipe **Tasks API** (the newest mediapipe interface; mediapipe>=0.10
removed the old `mp.solutions.face_detection`). Uses the BlazeFace short-range
detector. The model file (.tflite, with Tasks metadata) is auto-downloaded once
from Google's official model bucket and cached locally.

Robust fallback chain so preprocessing NEVER hard-fails:
    MediaPipe Tasks FaceDetector  ->  OpenCV Haar cascade  ->  center crop

A single FaceDetector is created lazily and reused across frames (fast, stable
box). Temporal smoothing: the previous box is kept when a frame has no detection,
which reduces ROI jitter (a real rPPG noise source).
"""
import os
os.environ.setdefault("GLOG_minloglevel", "3")   # silence MediaPipe clearcut/telemetry log spam
os.environ.setdefault("GLOG_logtostderr", "0")
import urllib.request
import numpy as np
import cv2

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
              "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_assets")
_MODEL_PATH = os.path.join(_MODEL_DIR, "blaze_face_short_range.tflite")


def _ensure_model():
    """Download the Tasks-API BlazeFace model once; return path or None."""
    if os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 50000:
        return _MODEL_PATH
    os.makedirs(_MODEL_DIR, exist_ok=True)
    try:
        import socket
        socket.setdefaulttimeout(15)
        print(f"[face] downloading BlazeFace model -> {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        if os.path.getsize(_MODEL_PATH) > 50000:
            return _MODEL_PATH
    except Exception as e:
        print(f"[face] model download failed ({e}); will use fallback detector")
    return None


class FaceCropper:
    """Detect + crop the face to a square `size`x`size` BGR image."""

    def __init__(self, size=72, expand=1.4, min_conf=0.5, backend="auto"):
        self.size = size
        self.expand = expand           # enlarge the box to include forehead/cheeks
        self.min_conf = min_conf
        self.prev_box = None           # (x, y, w, h) for temporal smoothing
        self.backend = None
        self._mp_det = None
        self._haar = None
        self._init_backend(backend)

    # ---------- backend setup ----------
    def _init_backend(self, backend):
        if backend in ("auto", "mediapipe"):
            if self._try_mediapipe():
                self.backend = "mediapipe"
                print("[face] backend = MediaPipe Tasks (BlazeFace short-range)")
                return
            if backend == "mediapipe":
                print("[face] MediaPipe requested but unavailable -> falling back")
        if self._try_haar():
            self.backend = "haar"
            print("[face] backend = OpenCV Haar cascade")
            return
        self.backend = "center"
        print("[face] backend = center-crop (no detector available)")

    def _try_mediapipe(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except Exception:
            return False
        model = _ensure_model()
        if model is None:
            return False
        try:
            opts = vision.FaceDetectorOptions(
                base_options=python.BaseOptions(model_asset_path=model),
                min_detection_confidence=self.min_conf)
            self._mp_det = vision.FaceDetector.create_from_options(opts)
            self._mp = mp
            return True
        except Exception as e:
            print(f"[face] MediaPipe init failed: {e}")
            return False

    def _try_haar(self):
        try:
            path = None
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                cand = os.path.join(cv2.data.haarcascades,
                                    "haarcascade_frontalface_default.xml")
                if os.path.exists(cand):
                    path = cand
            if path is None:
                import glob
                hits = glob.glob(os.path.join(os.path.dirname(cv2.__file__),
                                              "**", "haarcascade_frontalface_default.xml"),
                                 recursive=True)
                path = hits[0] if hits else None
            if path is None:
                return False
            self._haar = cv2.CascadeClassifier(path)
            return not self._haar.empty()
        except Exception:
            return False

    # ---------- detection ----------
    def _detect_box(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if self.backend == "mediapipe":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            res = self._mp_det.detect(mp_img)
            if res.detections:
                # pick the largest detection
                best = max(res.detections,
                           key=lambda d: d.bounding_box.width * d.bounding_box.height)
                b = best.bounding_box
                return (b.origin_x, b.origin_y, b.width, b.height)
            return None
        if self.backend == "haar":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            if len(faces):
                return tuple(max(faces, key=lambda f: f[2] * f[3]))
            return None
        return None  # center backend

    def crop(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        box = self._detect_box(frame_bgr)
        if box is None:
            box = self.prev_box        # reuse last good box (temporal smoothing)
        else:
            self.prev_box = box
        if box is None:
            return self._center(frame_bgr)
        x, y, bw, bh = box
        cx, cy = x + bw / 2, y + bh / 2
        side = max(bw, bh) * self.expand
        x0 = int(max(0, cx - side / 2)); y0 = int(max(0, cy - side / 2))
        x1 = int(min(w, cx + side / 2)); y1 = int(min(h, cy + side / 2))
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return self._center(frame_bgr)
        return cv2.resize(roi, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def _center(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return cv2.resize(frame_bgr[y0:y0 + s, x0:x0 + s], (self.size, self.size),
                          interpolation=cv2.INTER_AREA)

    def close(self):
        if self._mp_det is not None:
            try:
                self._mp_det.close()
            except Exception:
                pass

import os
import cv2
import time
import pickle
import numpy as np
from collections import deque
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import onnxruntime as ort
import torch
import csv

ort.preload_dlls()

# ================= CONFIG =================
CAMERA_ID = 0
ANTI_REAL = 1
ANTI_FAKE = 0
FIREBASE_URL = "https://hptproject-default-rtdb.asia-southeast1.firebasedatabase.app"

# One-shot: giảm window xuống còn ~1-2 giây (30fps x 1.5s = 45 frames)
ANTI_WINDOW = 15   # YOLO voting window (~0.5s)
FACE_WINDOW = 10   # FaceID voting window (~0.3s)

REAL_CONF_TH = 0.6
FAKE_CONF_TH = 0.6
FAKE_RATIO_TH_ANTI = 0.5
REAL_RATIO_TH_ANTI = 0.5
SUSPICIOUS_RATIO_TH = 0.4

# MiniFAS: dùng softmax index 0 = real (sau khi convert từ pth)
MINIFAS_REAL_IDX = 2   # thử index 0 trước, nếu vẫn sai đổi sang 1 hoặc 2
MINIFAS_TH = 999      # ngưỡng thấp hơn vì model convert không hoàn hảo


class MiniFASDetector:
    """
    Load 2 file onnx đã convert từ pth.
    Tự động tìm index class có score cao nhất để quyết định real/fake.
    """
    MODEL_FILES = [
        "2.7_80x80_MiniFASNetV2.onnx",
        "4_0_0_80x80_MiniFASNetV1SE.onnx",
    ]

    def __init__(self, model_dir, use_gpu=True):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu else ["CPUExecutionProvider"]
        )
        self.sessions = []
        for fname in self.MODEL_FILES:
            path = os.path.join(model_dir, fname)
            if os.path.exists(path):
                sess = ort.InferenceSession(path, providers=providers)
                self.sessions.append(sess)
                print(f"[MiniFAS] Loaded: {fname}")
            else:
                print(f"[MiniFAS] SKIP (not found): {fname}")

        self.enabled = False
        print(f"[MiniFAS] enabled={self.enabled}, models={len(self.sessions)}")

    def _preprocess(self, img, size=(80, 80)):
        img = cv2.resize(img, size)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        return img

    def predict(self, face_crop):
        """
        Trả về (is_real: bool, score: float).
        Score là xác suất class có index MINIFAS_REAL_IDX.
        """
        if not self.enabled or face_crop is None or face_crop.size == 0:
            return True, 1.0  # disabled → pass through

        inp = self._preprocess(face_crop)
        scores_list = []
        for sess in self.sessions:
            input_name = sess.get_inputs()[0].name
            out = sess.run(None, {input_name: inp})[0]  # shape: (1, N)
            # Softmax để normalize
            ex = np.exp(out[0] - np.max(out[0]))
            prob = ex / ex.sum()
            scores_list.append(prob)

        avg_prob = np.mean(scores_list, axis=0)
        real_score = float(avg_prob[MINIFAS_REAL_IDX])
        is_real = real_score >= MINIFAS_TH
        return is_real, real_score


class FaceIDEngine:
    def __init__(self, cam_id=CAMERA_ID):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        anti_model_path = os.path.join(base_dir, "phat.pt")
        face_db_path    = os.path.join(base_dir, "face_db.pkl")

        # --- PAD LOG ---
        self.LOG_PATH = os.path.join(base_dir, "pad_scores.csv")
        self._log_f = open(self.LOG_PATH, "a", newline="", encoding="utf-8")
        self._log_w = csv.writer(self._log_f)
        if self._log_f.tell() == 0:
            self._log_w.writerow(["ts", "score_bona", "real_ratio", "fake_ratio",
                                   "sus_ratio", "minifas_score", "state_end", "gt_label"])
        print("USING faceid_stream from:", os.path.abspath(__file__))

        # --- Camera ---
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera id={cam_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # --- GPU check ---
        assert torch.cuda.is_available(), "Torch CUDA NOT available!"
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        assert "CUDAExecutionProvider" in ort.get_available_providers(), \
            "ONNXRuntime khong co CUDA EP!"

        # --- YOLO (Tang 1) ---
        self.anti_model = YOLO(anti_model_path)

        # --- InsightFace ArcFace ---
        self.arcface = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.arcface.prepare(ctx_id=0)

        # --- MiniFAS (Tang 2) ---
        minifas_dir = os.path.join(base_dir, "anti_spoof_models")
        self.minifas = MiniFASDetector(model_dir=minifas_dir, use_gpu=True)
        self.minifas_scores = deque(maxlen=ANTI_WINDOW)

        # --- Face DB ---
        with open(face_db_path, "rb") as f:
            self.face_db = pickle.load(f)

        # --- State machine ---
        self.state = "IDLE"
        self.fake_frames       = deque(maxlen=ANTI_WINDOW)
        self.real_frames       = deque(maxlen=ANTI_WINDOW)
        self.suspicious_frames = deque(maxlen=ANTI_WINDOW)
        self.face_frames       = deque(maxlen=FACE_WINDOW)
        self.show_name_until     = 0
        self.fake_cooldown_until = 0
        self.faceid_fake_since   = None
        self.pending_attendance  = None

        print("Torch:", torch.__version__)
        print("GPU:", torch.cuda.get_device_name(0))

    def get_roi(self, frame):
        h, w = frame.shape[:2]
        box = int(min(w, h) * 0.45)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - box // 2, cy - box // 2
        x2, y2 = cx + box // 2, cy + box // 2
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def recognize(self, roi):
        faces = self.arcface.get(roi)
        if not faces:
            return None
        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)
        names = self.face_db["names"]
        embs  = self.face_db["embeddings"]
        best_name, best_sim = None, 0
        for i in range(len(names)):
            sim = np.dot(emb, embs[i])
            if sim > best_sim:
                best_sim, best_name = sim, names[i]
        return best_name if best_sim > 0.45 else None

    def reset(self):
        self.state = "IDLE"
        self.fake_frames.clear()
        self.real_frames.clear()
        self.suspicious_frames.clear()
        self.face_frames.clear()
        self.minifas_scores.clear()

    def log_presentation(self, real_ratio, fake_ratio, sus_ratio, minifas_avg, state_end):
        self._log_w.writerow([time.time(), real_ratio, real_ratio, fake_ratio,
                               sus_ratio, minifas_avg, state_end, ""])
        self._log_f.flush()

    def process(self, frame):
        now = time.time()
        roi, (x1, y1, x2, y2) = self.get_roi(frame)

        # --- FAKE COOLDOWN ---
        if now < self.fake_cooldown_until:
            cv2.putText(frame, "FAKE DETECTED - REMOVE SPOOF",
                        (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        # ===== TANG 1: YOLO =====
        res = self.anti_model.predict(frame, conf=0.3, device=0, verbose=False)[0]
        real_this_frame       = False
        fake_this_frame       = False
        suspicious_this_frame = False

        if res.boxes is not None:
            for box in res.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == ANTI_FAKE and conf >= FAKE_CONF_TH:
                    fake_this_frame = True
                elif cls == ANTI_REAL and conf >= REAL_CONF_TH:
                    real_this_frame = True

        # ===== TANG 2: MiniFAS =====
        minifas_real, minifas_score = self.minifas.predict(roi)
        self.minifas_scores.append(minifas_score)
        minifas_avg = float(np.mean(self.minifas_scores))

        # Ket hop: YOLO real + MiniFAS real → real
        if real_this_frame and not minifas_real:
            real_this_frame = False
            fake_this_frame = True
        
        if not fake_this_frame and not real_this_frame:
            suspicious_this_frame = True

        # ===== WINDOWS =====
        self.fake_frames.append(fake_this_frame)
        self.real_frames.append(real_this_frame)
        self.suspicious_frames.append(suspicious_this_frame)

        fake_ratio       = sum(self.fake_frames)       / len(self.fake_frames)
        real_ratio       = sum(self.real_frames)       / len(self.real_frames)
        suspicious_ratio = sum(self.suspicious_frames) / len(self.suspicious_frames)

        face_in_roi = True if self.arcface.get(roi) else False

        # ===== STATE MACHINE =====
        if self.state == "IDLE":
            if (face_in_roi
                    and fake_ratio       <  FAKE_RATIO_TH_ANTI
                    and real_ratio       >= REAL_RATIO_TH_ANTI
                    and suspicious_ratio <  SUSPICIOUS_RATIO_TH):
                self.state = "ANTI"
                self.fake_frames.clear()
                self.real_frames.clear()
                self.suspicious_frames.clear()

        elif self.state == "ANTI":
            if fake_ratio >= FAKE_RATIO_TH_ANTI or suspicious_ratio >= SUSPICIOUS_RATIO_TH:
                self.log_presentation(real_ratio, fake_ratio, suspicious_ratio, minifas_avg, "ANTI_FAIL")
                self.reset()
                self.fake_cooldown_until = now + 1.0
            elif len(self.fake_frames) == ANTI_WINDOW:
                self.log_presentation(real_ratio, fake_ratio, suspicious_ratio, minifas_avg, "ANTI_OK")
                self.state = "FACEID"
                self.face_frames.clear()

        elif self.state == "FACEID":
            if fake_this_frame or suspicious_this_frame:
                if self.faceid_fake_since is None:
                    self.faceid_fake_since = now
                elif now - self.faceid_fake_since >= 0.5:
                    self.log_presentation(real_ratio, fake_ratio, suspicious_ratio, minifas_avg, "FACEID_FAIL")
                    self.reset()
                    self.fake_cooldown_until = now + 1.5
                    self.faceid_fake_since = None
                    return frame
            else:
                self.faceid_fake_since = None

            name = self.recognize(roi)
            self.face_frames.append(name)

            if len(self.face_frames) == FACE_WINDOW:
                most  = max(set(self.face_frames), key=self.face_frames.count)
                ratio = self.face_frames.count(most) / FACE_WINDOW
                if most and ratio >= 0.7:
                    self.pending_attendance = {
                        "name":     most,
                        "date":     time.strftime("%Y-%m-%d"),
                        "time":     time.strftime("%H:%M:%S"),
                        "terminal": "K-04"
                    }
                    self.state = "SUCCESS"
                    self.show_name_until = now + 999
                else:
                    self.reset()

        elif self.state == "SUCCESS":
            if now > self.show_name_until:
                self.reset()

        # ===== OVERLAY =====
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 170), (360, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        state_color = {
            "IDLE": (255, 255, 0),
            "ANTI": (0, 165, 255),
            "FACEID": (0, 255, 0),
            "SUCCESS": (0, 255, 0),
        }.get(self.state, (255, 255, 255))

        cv2.putText(frame, f"STATE: {self.state}",
                    (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        cv2.putText(frame, f"REAL:{real_ratio:.2f} FAKE:{fake_ratio:.2f} SUS:{suspicious_ratio:.2f}",
                    (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        minifas_color = (0, 255, 0) if minifas_real else (0, 0, 255)
        cv2.putText(frame, f"MiniFAS:{minifas_avg:.2f} ({'REAL' if minifas_real else 'FAKE'})",
                    (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, minifas_color, 2)

        return frame
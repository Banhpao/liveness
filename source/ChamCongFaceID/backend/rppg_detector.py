import numpy as np
from collections import deque
import cv2

class RPPGDetector:
    def __init__(self, fps=30, window_sec=5):
        self.fps = fps
        self.window = int(fps * window_sec)
        self.g_buffer = deque(maxlen=self.window)
        self.r_buffer = deque(maxlen=self.window)
        self.b_buffer = deque(maxlen=self.window)

    def update(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return
        self.b_buffer.append(float(np.mean(face_roi[:, :, 0])))
        self.g_buffer.append(float(np.mean(face_roi[:, :, 1])))
        self.r_buffer.append(float(np.mean(face_roi[:, :, 2])))

    def compute_score(self):
        if len(self.g_buffer) < self.fps * 2:
            return {"score": 0.5, "bpm": 0, "ready": False}
        g = np.array(self.g_buffer)
        r = np.array(self.r_buffer)
        b = np.array(self.b_buffer)
        g_n = (g - np.mean(g)) / (np.std(g) + 1e-8)
        r_n = (r - np.mean(r)) / (np.std(r) + 1e-8)
        b_n = (b - np.mean(b)) / (np.std(b) + 1e-8)
        X = 3 * r_n - 2 * g_n
        Y = 1.5 * r_n + g_n - 1.5 * b_n
        alpha = np.std(X) / (np.std(Y) + 1e-8)
        pulse = X - alpha * Y
        n = len(pulse)
        freqs = np.fft.rfftfreq(n, d=1.0/self.fps)
        fft_mag = np.abs(np.fft.rfft(pulse * np.hanning(n)))
        valid = (freqs >= 0.75) & (freqs <= 3.0)
        if not np.any(valid):
            return {"score": 0.0, "bpm": 0, "ready": True}
        heart_power = np.sum(fft_mag[valid] ** 2)
        total_power = np.sum(fft_mag ** 2) + 1e-8
        power_ratio = heart_power / total_power
        peak_freq = freqs[valid][np.argmax(fft_mag[valid])]
        bpm = peak_freq * 60
        bpm_ok = 1.0 if 50 <= bpm <= 120 else 0.3
        score = float(np.clip(power_ratio * 3 * bpm_ok, 0, 1))
        return {"score": score, "bpm": round(bpm, 1), "ready": True}

    def reset(self):
        self.g_buffer.clear()
        self.r_buffer.clear()
        self.b_buffer.clear()

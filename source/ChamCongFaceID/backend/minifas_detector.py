import cv2
import numpy as np
import onnxruntime as ort
import os

class MiniFASDetector:
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
            if not os.path.exists(path):
                raise FileNotFoundError(f"Khong tim thay model: {path}")
            sess = ort.InferenceSession(path, providers=providers)
            self.sessions.append(sess)
        print(f"[MiniFAS] Loaded {len(self.sessions)} models, provider={providers[0]}")

    def _preprocess(self, img, size=(80, 80)):
        img = cv2.resize(img, size)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        return img

    def predict(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return 0.0
        inp = self._preprocess(face_crop)
        scores = []
        for sess in self.sessions:
            input_name = sess.get_inputs()[0].name
            out = sess.run(None, {input_name: inp})[0]
            score = float(out[0, 1])
            scores.append(score)
        return float(np.mean(scores))

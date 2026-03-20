import cv2
import time
import pickle
import numpy as np
import pyrealsense2 as rs
from collections import deque, Counter
from insightface.app import FaceAnalysis

# ================= CONFIG =================
ANTI_TIME_REQUIRED = 2.0
FACEID_FAKE_TIME = 1.0
SUCCESS_SHOW_TIME = 2.0

FACE_WINDOW = 30

Z_MIN = 200
Z_MAX = 800
NOSE_CLUSTER_RANGE = 120

STD_PHONE_TH = 7.0
STD_FACE_TH  = 20.0

COVERAGE_PAPER_TH = 0.75
SPATIAL_VAR_TH = 2.0
PLANE_RES_TH = 2.5   
FG_RANGE = 40          
FG_RATIO_TH = 0.35     # >35% foreground là đáng ngờ

# ================= ENGINE =================
class FaceID3DAntiSpoof:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        self.arcface = FaceAnalysis(name="buffalo_l")
        self.arcface.prepare(ctx_id=0)

        with open("face_db.pkl", "rb") as f:
            self.face_db = pickle.load(f)

        self.state = "IDLE"
        self.anti_start_time = None
        self.faceid_fake_since = None
        self.success_until = None

        self.anti_labels = []
        self.face_names = deque(maxlen=FACE_WINDOW)

    # ================= ROI =================
    def get_roi(self, frame, depth):
        h, w = frame.shape[:2]
        box = int(min(w, h) * 0.45)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - box // 2, cy - box // 2
        x2, y2 = cx + box // 2, cy + box // 2
        return frame[y1:y2, x1:x2], depth[y1:y2, x1:x2], (x1, y1, x2, y2)

    # ================= PLANE RESIDUAL =================
    def plane_residual(self, mask, depth_roi):
        ys, xs = np.where(mask)
        if len(xs) < 200:
            return 0.0

        zs = depth_roi[ys, xs]
        A = np.column_stack([xs, ys, np.ones_like(xs)])
        C, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)
        z_fit = A @ C
        return float(np.mean(np.abs(zs - z_fit)))

    # ================= FACE CLUSTER =================
    def extract_cluster(self, depth_roi):
        valid = depth_roi[(depth_roi >= Z_MIN) & (depth_roi <= Z_MAX)]
        if valid.size < 800:
            return None

        z_min = np.min(valid)

        # ===== foreground mask =====
        fg_mask = (depth_roi >= z_min) & (depth_roi <= z_min + FG_RANGE)
        fg_count = np.count_nonzero(fg_mask)

        # ===== face cluster mask =====
        cluster_mask = (depth_roi >= z_min) & (depth_roi <= z_min + NOSE_CLUSTER_RANGE)
        cluster = depth_roi[cluster_mask]
        if cluster.size < 800:
            return None

        h, w = depth_roi.shape
        coverage = cluster.size / (h * w)
        foreground_ratio = fg_count / np.count_nonzero(
            (depth_roi >= Z_MIN) & (depth_roi <= Z_MAX)
        )

        # ===== spatial variance (giữ nguyên của bạn) =====
        mid_h, mid_w = h // 2, w // 2
        stds = []
        for ys, ye in [(0, mid_h), (mid_h, h)]:
            for xs, xe in [(0, mid_w), (mid_w, w)]:
                part = depth_roi[ys:ye, xs:xe]
                part = part[(part >= z_min) & (part <= z_min + NOSE_CLUSTER_RANGE)]
                if part.size > 50:
                    stds.append(np.std(part))
        spatial_var = np.std(stds) if len(stds) >= 2 else 0

        plane_res = self.plane_residual(cluster_mask, depth_roi)

        return {
            "std": float(np.std(cluster)),
            "z_nose": float(z_min),
            "count": int(cluster.size),
            "coverage": coverage,
            "foreground_ratio": foreground_ratio,   # <<< MỚI
            "spatial_var": spatial_var,
            "plane_res": plane_res
        }


    # ================= DRAW =================
    def draw_face_cluster(self, frame, depth_roi, roi_pos, z_nose):
        x1, y1, _, _ = roi_pos
        h, w = depth_roi.shape
        for i in range(0, h, 3):
            for j in range(0, w, 3):
                z = depth_roi[i, j]
                if z_nose <= z <= z_nose + NOSE_CLUSTER_RANGE:
                    cv2.circle(frame, (x1 + j, y1 + i), 1, (0,255,0), -1)

    # ================= FACE RECOGNITION =================
    def recognize(self, roi):
        faces = self.arcface.get(roi)
        if not faces:
            return None

        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)

        best, best_sim = None, 0
        for name, ref in zip(self.face_db["names"], self.face_db["embeddings"]):
            sim = np.dot(emb, ref)
            if sim > best_sim:
                best, best_sim = name, sim

        return best if best_sim > 0.45 else None

    # ================= RESET =================
    def reset(self):
        self.state = "IDLE"
        self.anti_start_time = None
        self.faceid_fake_since = None
        self.success_until = None
        self.anti_labels.clear()
        self.face_names.clear()

    # ================= MAIN =================
    def run(self):
        while True:
            frames = self.align.process(self.pipeline.wait_for_frames())
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            frame = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data())
            now = time.time()

            roi, depth_roi, roi_pos = self.get_roi(frame, depth_img)
            cv2.rectangle(frame, roi_pos[:2], roi_pos[2:], (0,255,255), 2)

            stats = self.extract_cluster(depth_roi)
            label = "NONE"

            if stats:
                self.draw_face_cluster(frame, depth_roi, roi_pos, stats["z_nose"])
                std = stats["std"]

                if std < STD_PHONE_TH:
                    label = "FAKE"
                elif std > STD_FACE_TH:
                    label = "REAL"
                else:
                    # ===== vùng nguy hiểm 7–20 =====

                    # 0. Có vật thể chiếm ưu thế ở foreground → FAKE
                    if stats["foreground_ratio"] > FG_RATIO_TH:
                        label = "FAKE"

                    # 1. Bao phủ gần hết ROI → giấy / màn hình
                    elif stats["coverage"] > COVERAGE_PAPER_TH:
                        label = "FAKE"

                    # 2. Hình học gần phẳng
                    elif stats["plane_res"] < PLANE_RES_TH:
                        label = "FAKE"

                    # 3. Phân bố depth quá đều
                    elif stats["spatial_var"] < SPATIAL_VAR_TH:
                        label = "FAKE"

                    else:
                        label = "REAL"



            # ===== STATE MACHINE (GIỮ NGUYÊN) =====
            if self.state == "IDLE":
                if stats:
                    self.state = "ANTI"
                    self.anti_start_time = now
                    self.anti_labels.clear()

            elif self.state == "ANTI":
                self.anti_labels.append(label)
                if now - self.anti_start_time >= ANTI_TIME_REQUIRED:
                    if self.anti_labels.count("REAL") / len(self.anti_labels) >= 0.7:
                        self.state = "FACEID"
                        self.face_names.clear()
                        self.faceid_fake_since = None
                    else:
                        self.reset()

            elif self.state == "FACEID":
                if stats and label == "FAKE":
                    if self.faceid_fake_since is None:
                        self.faceid_fake_since = now
                    elif now - self.faceid_fake_since >= FACEID_FAKE_TIME:
                        self.reset()
                        continue
                else:
                    self.faceid_fake_since = None

                name = self.recognize(roi)
                self.face_names.append(name)

                if len(self.face_names) == FACE_WINDOW:
                    most, cnt = Counter(self.face_names).most_common(1)[0]
                    if most and cnt / FACE_WINDOW >= 0.8:
                        self.state = "SUCCESS"
                        self.success_until = now + SUCCESS_SHOW_TIME

            elif self.state == "SUCCESS":
                cv2.putText(frame, f"HELLO {most}", (50,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                if now > self.success_until:
                    self.reset()

            # ================= OVERLAY =================
            cv2.putText(frame, f"STATE: {self.state}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.putText(frame, f"LABEL: {label}", (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if stats:
                cv2.putText(frame, f"STD: {stats['std']:.2f}", (20,95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                cv2.putText(frame, f"coverage: {stats['coverage']:.2f}", (20,118),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                cv2.putText(frame, f"spatial var: {stats['spatial_var']:.2f}", (20,141),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                cv2.putText(frame, f"plane res: {stats['plane_res']:.2f}", (20,164),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                cv2.putText(frame, f"fg ratio: {stats['foreground_ratio']:.2f}",
                            (20,187),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)


            cv2.imshow("FaceID 3D", frame)
            if cv2.waitKey(1) == 27:
                break

        self.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    FaceID3DAntiSpoof().run()

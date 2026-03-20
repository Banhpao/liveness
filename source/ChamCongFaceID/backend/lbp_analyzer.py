import cv2
import numpy as np
from skimage.feature import local_binary_pattern

class LBPAnalyzer:
    """
    Phân tích texture bằng LBP để phân biệt da thật vs ảnh in.
    Nguyên lý: ảnh in/màn hình có texture patterns khác da người thật.
    """
    
    def __init__(self, n_points=24, radius=3, n_bins=64):
        self.n_points = n_points  # số điểm lân cận
        self.radius   = radius    # bán kính
        self.n_bins   = n_bins    # số bins histogram
        
        # Histogram chuẩn của da thật (cần calibrate với dữ liệu của bạn)
        # Đây là giá trị khởi tạo, sẽ được update qua collect_real_samples()
        self.real_hist_ref = None
    
    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        """Trích xuất LBP histogram từ vùng mặt."""
        if face_bgr is None or face_bgr.size == 0:
            return np.zeros(self.n_bins * 3)
        
        # Resize chuẩn
        face = cv2.resize(face_bgr, (128, 128))
        
        hists = []
        for ch in cv2.split(face):  # 3 kênh màu
            lbp = local_binary_pattern(
                ch, self.n_points, self.radius, method='uniform'
            )
            hist, _ = np.histogram(
                lbp.ravel(), bins=self.n_bins,
                range=(0, self.n_points + 2), density=True
            )
            hists.append(hist)
        
        return np.concatenate(hists)  # vector 192 chiều
    
    def compute_texture_score(self, face_bgr: np.ndarray) -> float:
        """
        Trả về score texture [0, 1].
        Cao = texture giống thật.
        Thấp = texture giống ảnh/màn hình.
        """
        if self.real_hist_ref is None:
            return 0.5  # chưa calibrate → neutral
        
        hist = self.extract(face_bgr)
        
        # Bhattacharyya similarity (càng giống histogram thật càng cao)
        bc = np.sum(np.sqrt(hist * self.real_hist_ref + 1e-10))
        return float(np.clip(bc, 0, 1))
    
    def collect_real_samples(self, face_list: list):
        """
        Calibrate bằng cách thu thập mẫu da thật.
        Gọi hàm này với ~50 frame mặt thật để tính histogram tham chiếu.
        """
        hists = [self.extract(f) for f in face_list if f is not None]
        if hists:
            self.real_hist_ref = np.mean(hists, axis=0)
            print(f"[LBP] Calibrated với {len(hists)} samples")
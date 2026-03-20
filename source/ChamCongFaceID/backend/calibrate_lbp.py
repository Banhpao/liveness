"""
Chạy script này 1 lần để calibrate LBP histogram.
Đứng trước camera ~30 giây để thu thập mẫu mặt thật.
"""
import cv2
import pickle
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lbp_analyzer import LBPAnalyzer

cap = cv2.VideoCapture(0)
analyzer = LBPAnalyzer()
samples = []

print("Thu thập mẫu mặt thật... Nhìn thẳng vào camera. Nhấn Q để dừng.")

while len(samples) < 100:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    roi = frame[h//4:3*h//4, w//4:3*w//4]
    samples.append(roi.copy())
    
    cv2.putText(frame, f"Samples: {len(samples)}/100", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Calibrate", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

analyzer.collect_real_samples(samples)

# Lưu histogram tham chiếu
with open("lbp_ref.pkl", "wb") as f:
    pickle.dump(analyzer.real_hist_ref, f)

print("Đã lưu lbp_ref.pkl")
cap.release()
cv2.destroyAllWindows()
import cv2
import os

# ================= CONFIG =================
DATASET_ROOT = r"D:\Source\ChamCongFaceID\backend\dataset"
CAMERA_ID = 0
IMG_SIZE = 256   # ảnh crop vuông

# ================= INPUT NAME =================
person_name = input("Nhập tên người cần thu thập ảnh: ").strip()
if not person_name:
    print("❌ Tên không hợp lệ")
    exit()

output_dir = os.path.join(DATASET_ROOT, person_name)
os.makedirs(output_dir, exist_ok=True)

print(f"[INFO] Lưu ảnh vào: {output_dir}")
print("▶ Đưa mặt vào khung")
print("▶ Nhấn SPACE để chụp | Q để thoát")

cap = cv2.VideoCapture(CAMERA_ID)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    annotated = frame.copy()

    # ===== GUIDE BOX (VUÔNG – CỐ ĐỊNH) =====
    box_size = int(min(w, h) * 0.45)
    cx, cy = w // 2, h // 2

    x1 = cx - box_size // 2
    y1 = cy - box_size // 2
    x2 = cx + box_size // 2
    y2 = cy + box_size // 2

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,255), 2)

    cv2.putText(
        annotated,
        f"{person_name} | {count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.imshow("BUILD DATA - ArcFace", annotated)
    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # SPACE
        face_crop = frame[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))

        filename = f"{str(count).zfill(5)}.png"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, face_crop)

        print(f"[INFO] Saved {path}")
        count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

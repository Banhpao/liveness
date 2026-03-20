import csv
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "pad_scores.csv")

def bpcer_apcer(y_true, score_bona, t):
    y_true = np.asarray(y_true).astype(int)      # 1=real(bona), 0=fake(attack)
    score_bona = np.asarray(score_bona).astype(float)
    pred_bona = score_bona >= t

    is_bona = (y_true == 1)
    is_attack = (y_true == 0)

    bpcer = np.mean(~pred_bona[is_bona]) if np.any(is_bona) else np.nan
    apcer = np.mean(pred_bona[is_attack]) if np.any(is_attack) else np.nan
    return bpcer, apcer

def sweep(y_true, score_bona, ts=np.linspace(0, 1, 1001)):
    rows = []
    for t in ts:
        bp, ap = bpcer_apcer(y_true, score_bona, t)
        rows.append((t, bp, ap))
    return rows

# đọc pad_scores.csv (log từ faceid_stream.py)
y_true = []
scores = []
pai = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "pad_scores.csv")
print("Reading:", csv_path)

with open(r"d:\source\source\ChamCongFaceID\backend\pad_scores.csv", "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["gt_label"] == "":      # bỏ những dòng realtime chưa có label
            continue
        # bạn có thể chỉ lấy ANTI_OK/ANTI_FAIL hoặc lấy riêng ANTI_OK
        y_true.append(int(row["gt_label"]))
        scores.append(float(row["score_bona"]))
        pai.append(row.get("pai",""))

rows = sweep(y_true, scores)

# ví dụ: chọn ngưỡng sao cho APCER <= 1%
target_ap = 0.01
best = None
for t, bp, ap in rows:
    if ap <= target_ap and not np.isnan(bp):
        if best is None or bp < best[1]:
            best = (t, bp, ap)

print("Best threshold for APCER<=1%:", best)

# nếu muốn APCER theo từng PAI species:
# group scores theo 'pai' rồi tính apcer riêng từng nhóm

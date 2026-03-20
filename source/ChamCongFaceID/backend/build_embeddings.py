import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
# ================= CONFIG =================
DATASET_ROOT = r"D:\Source\ChamCongFaceID\backend\dataset"
DB_PATH = r"D:\Source\ChamCongFaceID\backend\face_db.pkl"
INPUT_SIZE = (112,112)

# ================= LOAD DB =================
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
else:
    db = {"names": [], "embeddings": np.empty((0,512))}

existing_people = set(db["names"])
print(f"[INFO] Existing identities: {len(existing_people)}")

# ================= LOAD ARCFACE =================
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640,640))
rec_model = app.models["recognition"]

# ================= SCAN DATASET =================
new_people = [
    p for p in os.listdir(DATASET_ROOT)
    if os.path.isdir(os.path.join(DATASET_ROOT, p)) and p not in existing_people
]

if not new_people:
    print("[INFO] No new identities found")
    exit()

print("[INFO] New identities:", new_people)

# ================= BUILD EMBEDDINGS =================
for person in new_people:
    person_dir = os.path.join(DATASET_ROOT, person)
    count = 0

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        face = cv2.resize(img, INPUT_SIZE)
        emb = rec_model.get_feat([face])[0]
        emb = emb / np.linalg.norm(emb)

        db["names"].append(person)
        db["embeddings"] = np.vstack([db["embeddings"], emb])
        count += 1

    print(f"[OK] Added {count} embeddings for {person}")

# ================= SAVE DB =================
with open(DB_PATH, "wb") as f:
    pickle.dump(db, f)

print("[DONE] Face DB updated")

import os
from multiprocessing import freeze_support
from ultralytics import YOLO

def main():
    model_path = r"D:\source\source\ChamCongFaceID\backend\phat.pt"
    data_yaml  = r"D:\source\source\ChamCongFaceID\backend\dataset\Fake or real.v6i.yolov8\data.yaml"
    runs_dir   = r"D:\source\source\ChamCongFaceID\runs"

    print("model exists =", os.path.exists(model_path))
    print("data  exists =", os.path.exists(data_yaml))
    print("runs dir     =", runs_dir)

    model = YOLO(model_path)
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,      
        workers=0,     # QUAN TRỌNG: Windows spawn => để 0 cho chắc
        project=runs_dir,
        name="phat_1",
        exist_ok=True
    )

if __name__ == "__main__":
    freeze_support()
    main()
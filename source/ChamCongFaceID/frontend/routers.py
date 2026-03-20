import cv2
from flask import Response
from backend.faceid_stream import FaceIDEngine

engine = FaceIDEngine()


def gen_frames():
    while True:
        ret, frame = engine.cap.read()
        if not ret:
            break

        frame = engine.process(frame)
        _, buffer = cv2.imencode(".jpg", frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


def video():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

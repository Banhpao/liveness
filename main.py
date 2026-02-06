import os
import requests
from flask import Flask, render_template

from frontend.routers import video, engine
from backend.faceid_stream import FIREBASE_URL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "frontend", "templates")
)

@app.route("/")
def index():
    return render_template("index.html")

app.add_url_rule("/video", "video", video)

@app.route("/api/commit", methods=["POST"])
def commit_attendance():
    if engine.pending_attendance is None:
        return {"ok": False}

    data = engine.pending_attendance
    engine.pending_attendance = None

    date = data["date"]
    name = data["name"]

    url = f"{FIREBASE_URL}/attendance/{date}/{name}.json"

    requests.put(url, json={
        "time": data["time"],
        "terminal": data["terminal"]
    })

    engine.reset()
    return {"ok": True}

@app.route("/api/pending")
def get_pending():
    if engine.pending_attendance is None:
        return {"ok": False}
    return {
        "ok": True,
        "data": engine.pending_attendance
    }

@app.route("/api/reset", methods=["POST"])
def reset_attendance():
    engine.pending_attendance = None
    engine.reset()
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)

"""Local, dry-run GestureFlow dashboard."""
import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from actions import ActionController, dry_run_backend
from features import landmarks_to_features
from smoothing import GestureStabilizer

ROOT = Path(__file__).resolve().parent
LABELS = ["Open", "Close", "Pointer", "Ok", "Nice"]
cfg = json.loads((ROOT / "config.json").read_text())
model = joblib.load(ROOT / "models/best_model.joblib")
vision = mp.tasks.vision
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


def detector(mode):
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "models/hand_landmarker.task")),
        running_mode=mode, num_hands=1)
    return vision.HandLandmarker.create_from_options(options)


def mp_image(frame):
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def prediction(hand, frame, side):
    height, width = frame.shape[:2]
    features = landmarks_to_features(hand, width, height, mirror=side == "left")
    if features is None:
        return {}
    row = features.reshape(1, -1)
    if hasattr(model, "feature_names_in_"):
        row = pd.DataFrame(row, columns=model.feature_names_in_)
    probabilities = {LABELS[int(c)]: float(p) for c, p in
                     zip(model.classes_, model.predict_proba(row)[0])}
    label = max(probabilities, key=probabilities.get)
    return dict(gesture=label, confidence=probabilities[label], probabilities=probabilities)


def placeholder(msg):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, msg, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return cv2.imencode(".jpg", frame)[1].tobytes()


class CameraWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.RLock()
        self.hand = "right"
        self.jpeg = placeholder("Starting camera...")
        self.state = dict(camera="starting", gesture=None, confidence=0,
                          stable=None, probabilities={}, fps=0, hand_visible=False)
        self.stabilizer = GestureStabilizer()
        self.controller = ActionController(cfg["mapping"], dry_run_backend,
                                           cfg["cooldown_seconds"], cfg["repeat_interval_seconds"])
        self.controller.log = deque(maxlen=20)

    def snapshot(self):
        with self.lock:
            actions = [dict(time=time.strftime("%H:%M:%S", time.localtime(t)), gesture=g,
                            action=a, mode="dry run") for t, g, a in reversed(self.controller.log)]
            return dict(self.state, hand=self.hand, actions=actions, live=False,
                        live_available=False, control_enabled=self.controller.enabled)

    def run(self):
        cap = cv2.VideoCapture(0)
        try:
            with detector(vision.RunningMode.VIDEO) as hands:
                last = time.monotonic()
                timestamp = 0
                fps = 0
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = cv2.flip(frame, 1)
                    now = time.monotonic()
                    timestamp = max(timestamp + 1, int(now * 1000))
                    result = hands.detect_for_video(mp_image(frame), timestamp)
                    hand = result.hand_landmarks[0] if result.hand_landmarks else []
                    h, w = frame.shape[:2]
                    for point in hand:
                        cv2.circle(frame, (int(point.x*w), int(point.y*h)), 4, (0, 0, 255), -1)
                    with self.lock:
                        info = prediction(hand, frame, self.hand) if hand else {}
                        label, conf = info.get("gesture"), info.get("confidence", 0)
                        stable, trigger = self.stabilizer.update(label, conf, now)
                        self.controller.update(stable, trigger, time.time())
                        fps = 0.9*fps + 0.1/max(now-last, 0.000001)
                        last = now
                        cv2.putText(frame, stable or "-", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 255), 2)
                        self.jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                        self.state.update(camera="ok", gesture=label, confidence=conf, stable=stable,
                                          probabilities=info.get("probabilities", {}),
                                          fps=round(fps, 1), hand_visible=bool(hand))
        finally:
            cap.release()
            with self.lock:
                self.jpeg = placeholder("Camera unavailable")
                self.state.update(camera="error", gesture=None, confidence=0, stable=None,
                                  probabilities={}, fps=0, hand_visible=False)


worker = CameraWorker()


@app.get("/")
def index():
    return render_template("index.html", labels=LABELS, mapping=cfg["mapping"])


@app.get("/video_feed")
def video_feed():
    def frames():
        while True:
            with worker.lock:
                jpeg = worker.jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.04)
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/state")
def state():
    return jsonify(worker.snapshot())


@app.post("/api/control")
def control():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400
    if "hand" in data and data["hand"] not in ("right", "left"):
        return jsonify(error="Choose right or left"), 400
    if any(type(data[k]) is not bool for k in ("live", "enabled") if k in data):
        return jsonify(error="Control switches require true or false"), 400
    if data.get("live"):
        return jsonify(error="Live control is unavailable in Step 20a"), 400
    with worker.lock:
        hand = data.get("hand", worker.hand)
        enabled = data.get("enabled", worker.controller.enabled)
        if hand != worker.hand or enabled != worker.controller.enabled:
            worker.stabilizer = GestureStabilizer()
            worker.state["stable"] = None
        worker.hand, worker.controller.enabled = hand, enabled
        return jsonify(worker.snapshot())


@app.post("/api/predict")
def predict():
    file = request.files.get("image")
    side = request.form.get("hand", "right")
    if side not in ("left", "right"):
        return jsonify(error="Choose right or left"), 400
    content = file.read() if file else b""
    if not content:
        return jsonify(error="Choose an image"), 400
    frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="Could not read the image"), 400
    with detector(vision.RunningMode.IMAGE) as hands:
        result = hands.detect(mp_image(frame))
    info = prediction(result.hand_landmarks[0], frame, side) if result.hand_landmarks else {}
    return jsonify(hand_found=bool(info), **info)


if __name__ == "__main__":
    worker.start()
    app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False)

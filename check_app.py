import io

import cv2
import numpy as np

from app import app

client = app.test_client()


def check(name, passed):
    assert passed, name
    print("PASS", name)


page = client.get("/")
check("home page loads", page.status_code == 200 and b"GestureFlow" in page.data)
state = client.get("/api/state").get_json()
check("state fields", {"stable", "gesture", "control_enabled", "live", "hand", "actions"} <= state.keys())
check("starts OFF, dry run, right hand", not state["control_enabled"] and not state["live"] and state["hand"] == "right")
state = client.post("/api/control", json={"enabled": True, "hand": "left"}).get_json()
check("control settings change", state["control_enabled"] and state["hand"] == "left")
check("LIVE refused in dry-run build", client.post("/api/control", json={"live": True}).status_code == 400)
check("invalid hand rejected", client.post("/api/control", json={"hand": "middle"}).status_code == 400)
client.post("/api/control", json={"enabled": False, "hand": "right"})
blank = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))[1].tobytes()
response = client.post("/api/predict", data={"image": (io.BytesIO(blank), "blank.jpg")})
check("blank image has no hand", response.status_code == 200 and response.json["hand_found"] is False)
response = client.post("/api/predict", data={"image": (io.BytesIO(b"invalid"), "bad.jpg")})
check("invalid image rejected", response.status_code == 400)
check("missing upload rejected", client.post("/api/predict").status_code == 400)

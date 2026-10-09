import os
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

ROOT = "data/raw/data/train"
CLASSES = ["fist", "five", "okay", "peace", "rad", "straight", "thumbs"]
N = 20

def make_landmarker(conf):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=conf,
    )
    return HandLandmarker.create_from_options(options)

def enlarge(img):
    # scale so the longest side is 512px, then add padding around the edges
    h, w = img.shape[:2]
    scale = 512 / max(h, w)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(img, 128, 128, 128, 128, cv2.BORDER_REPLICATE)

def detected(landmarker, img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return bool(landmarker.detect(mp_image).hand_landmarks)

# name: (confidence threshold, enlarge the image first?)
configs = {
    "original, conf 0.5": (0.5, False),
    "enlarged+padded, conf 0.5": (0.5, True),
    "enlarged+padded, conf 0.1": (0.1, True),
}
landmarkers = {name: make_landmarker(c) for name, (c, _) in configs.items()}

for label in CLASSES:
    folder = os.path.join(ROOT, label)
    files = sorted(os.listdir(folder))[:N]
    unreadable = 0
    shape = None
    hits = {name: 0 for name in configs}

    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            unreadable += 1
            continue
        shape = img.shape
        for name, (_, big) in configs.items():
            test_img = enlarge(img) if big else img
            if detected(landmarkers[name], test_img):
                hits[name] += 1

    print(f"{label}: image shape {shape}, unreadable {unreadable}/{len(files)}")
    for name, count in hits.items():
        print(f"    {name}: {count}/{len(files)} detected")
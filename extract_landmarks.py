import os
import cv2
import csv
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

DATA_ROOT="data/raw/data"
SPLITS=["train","validation","test"]
CLASSES=["fist","five","none","okay","peace","rad","straight","thumbs"]

options=HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)



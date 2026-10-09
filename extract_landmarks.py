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
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

#["split", "label",
#"x0", "y0", "z0",
#"x1", "y1", "z1",
#"x2", "y2", "z2",
# ...]

columns=["split","label"]
for i in range(21):
    columns+=[f"x{i}",f"y{i}",f"z{i}"]

rows=[]
skipped=0



with HandLandmarker.create_from_options(options) as landmarker:
    for split in SPLITS:
        for label in CLASSES:
            folder=os.path.join(DATA_ROOT,split,label)
            if not os.path.isdir(folder):
                print(f"Folder {folder} does not exist. Skipping...")
                continue

            files=os.listdir(folder)
            print(f"Processing {len(files)} files in {folder}")

            for fname in files:
                path=os.path.join(folder,fname)
                image=cv2.imread(path)

                if image is None:
                    skipped+=1
                    continue

                rgb= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

                result=landmarker.detect(mp_image)

                #Open 148.png.
                #Convert it to RGB.
                #Create mp_image.
                #Run landmarker.detect(mp_image).
                #Store the result in result.

                if not result.hand_landmarks:
                    skipped+=1
                    continue


                #Get the first detected hand.
                hand = result.hand_landmarks[0]
                row=[split,label]

                for lm in hand:
                    row+=[lm.x,lm.y,lm.z]
                rows.append(row)

print(f"Done. Collected {len(rows)} rows, "f"skipped {skipped} images (no hand detected or unreadable).")

with open("landmarks.csv","w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(columns)
    writer.writerows(rows)

print("Saved landmarks.csv")





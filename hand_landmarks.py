import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
#base options contains general configuration options for a MediaPipe model
HandLandmarker = mp.tasks.vision.HandLandmarker
#It is the thing that analyzes an image and finds the hand landmarks.
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
#This allows you to configure the hand landmarker.
RunningMode = mp.tasks.vision.RunningMode
#This gives you the available ways MediaPipe can process input.

options=HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

CONNECTIONS=[
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
]

cap=cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    timestamp_ms=0
    while True:
        ok,frame= cap.read()
        if not ok:
            break

        frame=cv2.flip(frame,1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Convert the image from BGR format to RGB format. as opencv gives us the image in BGR format but mediapipe needs it in RGB format.
        mp_image= mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        #mp.Image is a class that represents an image in MediaPipe. It takes two arguments: image_format and data. image_format specifies the format of the image, and data is the actual image data.

        timestamp_ms +=33
        #The timestamp_ms variable is used to keep track of the time in milliseconds. It is incremented by 33 milliseconds for each frame, which corresponds to a frame rate of approximately 30 frames per second (1000 ms / 30 fps ≈ 33 ms per frame). This timestamp is passed to the detect_for_video method to help the hand landmarker process the video frames in a time-aware manner.

        result = landmarker.detect_for_video(mp_image,timestamp_ms)

        if result.hand_landmarks:
            h, w, _ = frame.shape
            hand=result.hand_landmarks[0]

            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            for a,b in CONNECTIONS:
                cv2.line(frame,points[a],points[b],(0,255,0),2)
            for p in points:
                cv2.circle(frame, p, 4, (0,0,255), -1)

        cv2.imshow("GestureFlow - hand landmarks",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
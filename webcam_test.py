import cv2

cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    raise SystemExit

while True:
    ok, frame=cap.read()
    if not ok:
        break

    frame = cv2.flip(frame,1)
    cv2.imshow("GestureFlow- webcam test",frame)

    if cv2.waitKey(1)& 0xFF==ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
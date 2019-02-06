import cv2
import numpy as np
import sys

facePath = "haarcascade_frontalface_default.xml"
smilePath = "haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)
smileCascade = cv2.CascadeClassifier(smilePath)

cap = cv2.VideoCapture('smile_2.mp4')
cap.set(3, 640)
cap.set(4, 480)

sF = 1.05

while (cap.isOpened()):

    ret, frame = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=sF,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # ---- Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=2.0,
            minNeighbors=25,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
            )

        for (x1, y1, w1, h1) in smile:
            cv2.rectangle(roi_color, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 1)
            mag = round((w1/x), 2)
            mag = str(int(100*mag))
            print("Found a smile! and its magnitude is " + mag)
        print("Found a smile! and its magnitude is 0")

    cv2.imshow('Smile Detector', frame)
    c = cv2.waitKey(7) % 0x100
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
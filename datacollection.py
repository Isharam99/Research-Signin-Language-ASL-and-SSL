import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

def initialize_camera():
    cap = cv2.VideoCapture(0)  # 0 for webcam
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")
    return cap

cap = initialize_camera()
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

folder = "C:/Users/PCS/Desktop/1/Data/Z"  # path to save images

# data collection
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image. Restarting camera...")
        cap.release()
        time.sleep(1)  # Wait for a second before restarting the camera
        cap = initialize_camera()
        continue

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # bounding box

        # Ensure the coordinates are within the bounds of the image
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("Empty crop, skipping frame")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # white image

        aspectRatio = h / w  # aspect ratio of cropped image

        if aspectRatio > 1:  # height > width
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:  # width > height
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == ord("q"):
        break
        

cap.release()
cv2.destroyAllWindows()
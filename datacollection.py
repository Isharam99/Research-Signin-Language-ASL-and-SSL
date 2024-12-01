import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)                              # 0 for webcam
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

folder = "C:/Users/PCS/Desktop/1/Data Words/Hello"  # path to save images


# data collection
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # bounding box

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255  # white image

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]  # cropped image
        imgCropShape = imgCrop.shape  # cropped image shape

        aspectRatio = h / w  # aspect ratio of cropped image

        if aspectRatio > 1:   # height > width
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:  # width > height
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
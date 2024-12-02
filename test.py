import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D

print(tf.__version__)

# Load labels dynamically from the file
labels_path = "Model/labels.txt"
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Labels file not found: {labels_path}")

with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

# Load the model, ensuring compatibility with custom objects if needed
model_path = "Model/keras_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Custom DepthwiseConv2D layer to handle the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove the 'groups' argument if it exists
        super().__init__(*args, **kwargs)

# Custom object handling
custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}

try:
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    # Compile the model manually if needed for further training
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Custom classifier class to accept the loaded model
class CustomClassifier:
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels

    def getPrediction(self, img, draw=True):
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img)
        index = np.argmax(predictions)
        confidence = predictions[0][index]
        return predictions, index, confidence

classifier = CustomClassifier(model, labels)

offset = 20
imgSize = 300
counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    hands, img = detector.findHands(img)
    if hands:
        combined_label = ""
        for hand in hands:
            x, y, w, h = hand['bbox']

            # Ensure the coordinates are within the bounds of the image
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y1:y2, x1:x2]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index, confidence = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]
            combined_label += f"{label} ({confidence * 100:.2f}%) "

            # Draw bounding box around each hand
            cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Display the combined prediction on the image
        combined_label = combined_label.strip()
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 2
        font_thickness = 2
        text_size = cv2.getTextSize(combined_label, font, font_scale, font_thickness)[0]

        # Calculate the position for the rectangle and text
        text_x = 10
        text_y = 50
        rect_x1 = text_x - 10
        rect_y1 = text_y - text_size[1] - 10
        rect_x2 = text_x + text_size[0] + 20
        rect_y2 = text_y + 10

        # Display the prediction on the image
        cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, combined_label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
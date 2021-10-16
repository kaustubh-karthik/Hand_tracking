import cv2
from hand_tracking_utils import hand_detector as hands
import mediapipe as mp
import numpy as np

detector = hands()

cap = cv2.VideoCapture(1) # Check for error

while True:

    success, img = cap.read()

    img = detector.find_hands(img)
    lm_list = detector.find_lms(img, draw=True)

    if lm_list:
        print(lm_list[4])

    detector.display_img(img)
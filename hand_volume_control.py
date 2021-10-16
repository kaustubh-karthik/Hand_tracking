import cv2
from hand_tracking_utils import hand_detector
import math
import osascript

detector = hand_detector()

cap = cv2.VideoCapture(1) # Check for error

while True:

    success, img = cap.read()

    img = detector.find_hands(img)
    lm_list = detector.find_lms(img, )

    if lm_list:

        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]

        cv2.circle(img, (x1, y1), 10, (255, 0, 0))
        cv2.circle(img, (x2, y2), 10, (255, 0, 0))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255))

        dist_between = math.hypot(x1 - x2, y1 - y2)

        if dist_between > 300: dist_between = 300

        dist_percent = int(dist_between / 300 * 100)

        osascript.run(f"set volume output volume {dist_percent}")

    detector.display_img(img)

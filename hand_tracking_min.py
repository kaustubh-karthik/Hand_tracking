import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

prev_time = 0
curr_time = 0

while True:
    success, img = cap.read()

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_img)
    # print(results.multi_hand_landmarks)

    hand_landmark = results.multi_hand_landmarks

    if hand_landmark:

        for hand_lm in hand_landmark:

            for id, lm in enumerate(hand_lm.landmark):

                height, width, channels = img.shape
                centre_x, centre_y = int(lm.x * width), int(lm.y * height)

                print(id, ':', centre_x, centre_y)

                if id == 0:
                    cv2.circle(img, (centre_x, centre_y), 25, (255, 0, 255))


            mp_draw.draw_landmarks(img, hand_lm, mp_hands.HAND_CONNECTIONS)

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        img,
        text = str(int(fps)),
        org =(10, 70),
        fontFace = cv2.FONT_HERSHEY_COMPLEX,
        fontScale = 3,
        color = (150, 150, 150),
        thickness = 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
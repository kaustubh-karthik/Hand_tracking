import cv2
import mediapipe as mp
import numpy as np
import time
class hand_detector():
    def __init__(self, mode = False, max_hands = 2, detection_conf = 0.5, track_conf = 0.5):
        # Do i need variables? -- Check for static
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(mode, max_hands, detection_conf, track_conf)
        self.mp_draw = mp.solutions.drawing_utils

        self.prev_time = 0


    def find_hands(self, img, draw = True):
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(rgb_img)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:

            for hand_lm in self.results.multi_hand_landmarks:

                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lm, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_lms(self, img, hand_num = 0, draw = True):
        
        lm_list = []

        if self.results.multi_hand_landmarks:

            hand_lm = self.results.multi_hand_landmarks[hand_num]

            for id, lm in enumerate(hand_lm.landmark):

                height, width, channels = img.shape
                centre_x, centre_y = int(lm.x * width), int(lm.y * height)

                lm_list.append([id, centre_x, centre_y])

                if draw:
                    cv2.circle(img, (centre_x, centre_y), 10, (255, 0, 0))

        return lm_list

    def display_img(self, img, fps = True):

        if fps:
            curr_time = time.time()
            fps = 1/(curr_time - self.prev_time)
            self.prev_time = curr_time


        cv2.putText(
            img,
            text = str(int(fps)),
            org = (10, 70),
            fontFace = cv2.FONT_HERSHEY_COMPLEX,
            fontScale = 3,
            color = (150, 150, 150),
            thickness = 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


def main():

    detector = hand_detector()
    
    cap = cv2.VideoCapture(1) # Check for error

    while True:

        success, img = cap.read()

        img = detector.find_hands(img)
        lm_list = detector.find_lms(img)

        if lm_list:
            print(lm_list[4])

        detector.display_img(img)

if __name__ == '__main__':
    main()

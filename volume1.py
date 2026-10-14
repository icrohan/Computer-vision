import cv2
import mediapipe as mp
from math import hypot
import pyautogui
import numpy as np
from scipy.ndimage import gaussian_filter1d  
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils 


smooth_factor = 3  
previous_vol = 0    
vol_history = [0]   

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []  
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if lmList:
        
        x1, y1 = lmList[4][1], lmList[4][2]  
        x2, y2 = lmList[8][1], lmList[8][2]  

        cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)

        raw_vol = np.interp(length, [30, 300], [0, 100])

        vol_history.append(raw_vol)
        if len(vol_history) > 10:  
            vol_history.pop(0)
        smoothed_vol = gaussian_filter1d(vol_history, sigma=smooth_factor)[-1]

        if abs(smoothed_vol - previous_vol) > 2:  
            if smoothed_vol > previous_vol:
                pyautogui.press('volumeup')
            elif smoothed_vol < previous_vol:
                pyautogui.press('volumedown')
            previous_vol = smoothed_vol 

        cv2.putText(
            img, f'Volume: {int(smoothed_vol)}%', (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    cv2.imshow("Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()

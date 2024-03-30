import cv2
import mediapipe as mp
import numpy as np
import time


mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)


thumb_click_count = 0

partial_count = 0


def calculate_angle(a, b):
    a = np.array(a)  # Thumb MCP
    b = np.array(b)  # Thumb tip
    
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    return angle

# Setup MediaPipe instance for hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()

        # Make hand detection
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_mcp = [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
                             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y]  # Thumb MCP
                thumb_tip = [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y]  # Thumb tip
                angle = calculate_angle(thumb_mcp, thumb_tip)
                if angle < 12:
                    thumb_click_count += 1
                    print("Thumb click detected. Total clicks:", thumb_click_count)
                    time.sleep(0.3)
                elif 18 < angle < 21: 
                    partial_count += 1
                    print("Partial Clicks: ", partial_count)
                    time.sleep(0.7)
        cv2.putText(frame, f'Thumb Clicks: {thumb_click_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(frame, f'Partial Clicks: {partial_count}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        cv2.imshow('MediaPipe Hand Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

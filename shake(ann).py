import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

lower_color = np.array([100, 100, 100])
upper_color = np.array([140, 255, 255])

pen_shake_count = 0

prev_pen_tip_pos = None

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_color, upper_color)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Filter out small contours noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                pen_tip_center = (x + w // 2, y + h // 2)
                cv2.circle(frame, pen_tip_center, 5, (0, 255, 0), -1)
                break  # Consider only the largest contour as the pen tip

        # Track pen tip movement for shake detection
        if prev_pen_tip_pos is not None and 'pen_tip_center' in locals():
            prev_x, prev_y = prev_pen_tip_pos
            curr_x, curr_y = pen_tip_center
            displacement = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            if displacement > 30:  # threshold
                pen_shake_count += 1
                print("Pen shake detected. Total shakes:", pen_shake_count)

        # Update previous pen tip position
        prev_pen_tip_pos = pen_tip_center if 'pen_tip_center' in locals() else None

        # Render hand landmarks
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display pen shake count on the screen
        cv2.putText(frame, f'Pen Shakes: {pen_shake_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        cv2.imshow('Hand and Pen Tip Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

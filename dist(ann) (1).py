import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def main():
    cap = cv2.VideoCapture(0)  # Use the camera
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect the pose
            results = pose.process(image_rgb)
            
            # Extract nose and right hand keypoints
            if results.pose_landmarks:
                nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                right_hand_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]

                # Draw connections and landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Calculate distance between nose and right hand
                distance = calculate_distance(nose_landmark, right_hand_landmark)
                cv2.putText(frame, f'Distance: {distance:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Show the frame
            cv2.imshow('Frame', frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

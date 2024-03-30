from flask import Flask, Response, render_template
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe components
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Generator function for webcam video frames
def generate_frames():
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection, \
         mp_face_mesh.FaceMesh(
             min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
         mp_pose.Pose(
             min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(
             min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Image processing
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results_detection = face_detection.process(image)
            results_mesh = face_mesh.process(image)
            results_pose = pose.process(image)
            results_hands = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results_detection.detections:
                for detection in results_detection.detections:
                    mp_drawing.draw_detection(image, detection)

            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,0), thickness=1))
            
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,255,0), thickness=2))
            
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,255), thickness=2))

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route to stream webcam video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to render HTML page with video stream
@app.route('/')
def index():
    return render_template('index.html')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)

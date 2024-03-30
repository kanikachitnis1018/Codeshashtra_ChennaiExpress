from flask import render_template, Response
from flask_login import login_user
from package.forms import LoginForm, RegisterForm
from package.models import User
from package import app, db
import cv2

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/guide")
def guide():
    return render_template("guide.html")

# Function to annotate frames
def annotate_frame(frame, face_landmarks, pose_landmarks, hand_landmarks):
    # Draw landmarks on detected faces
    if face_landmarks:
        for face_landmarks in face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

    # Draw landmarks on detected poses
    if pose_landmarks:
        for landmark in pose_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

    # Draw landmarks on detected hands
    if hand_landmarks:
        for hand_landmarks in hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (255, 0, 255), 1)

    return frame

# Generator function for webcam video frames with annotations
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

            annotated_frame = annotate_frame(image, results_detection, results_mesh, results_pose, results_hands)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route("/analyse")
def analyse():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/login", methods=['GET','POST'])
def login():
    form = LoginForm()
    return render_template("login.html", form=form)

@app.route("/register", methods=['GET','POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(username=form.username.data,
                              email=form.email.data,
                              password=form.password1.data)
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)

    return render_template("register.html", form=form)

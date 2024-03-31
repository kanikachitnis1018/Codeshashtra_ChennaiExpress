from flask import render_template, redirect, url_for, flash, request, jsonify, Response, requests
from flask_login import login_required, login_user, logout_user
from package.forms import LoginForm, RegisterForm
from package.models import User
from package import app, db
import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pen_shake_count = 0 

def calculate_angle(a, b):
    a = np.array(a)  # Thumb MCP
    b = np.array(b)  # Thumb tip
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle

# Function to perform hand detection
def hand_detection():
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    thumb_click_count = 0
    partial_count = 0

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

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def detect_hand_and_pen():
    cap = cv2.VideoCapture(0)
    
    lower_color = np.array([100, 100, 100])
    upper_color = np.array([140, 255, 255])
    
    prev_pen_tip_pos = None

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_color, upper_color)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    pen_tip_center = (x + w // 2, y + h // 2)
                    cv2.circle(frame, pen_tip_center, 5, (0, 255, 0), -1)
                    break  

            if prev_pen_tip_pos is not None and 'pen_tip_center' in locals():
                prev_x, prev_y = prev_pen_tip_pos
                curr_x, curr_y = pen_tip_center
                displacement = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                if displacement > 30:  
                    pen_shake_count += 1

            prev_pen_tip_pos = pen_tip_center if 'pen_tip_center' in locals() else None

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f'Pen Shakes: {pen_shake_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

mp_pose = mp.solutions.pose

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def classify_distance(distance):
    if distance <= 0.3:
        return "Perfect"
    elif 0.3 < distance <= 1:
        return "Almost there"
    else:
        return "Too Far"

def pose_detection():
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

                # Calculate distance between nose and right hand
                distance = calculate_distance(nose_landmark, right_hand_landmark)
                # Classify the distance
                classification = classify_distance(distance)
                cv2.putText(frame, f'Distance: ({classification})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Redirect if classification is "Perfect"
                if classification == "Perfect":
                    cap.release()
                    cv2.destroyAllWindows()
                    return redirect(url_for('click'))

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as a byte string
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/guide")
# @login_required
def guide():
    return render_template("guide.html")

@app.route("/shake")
#@login_required
def shake():
    global pen_shake_count
    if pen_shake_count > 15:
        return redirect(url_for('distance'))
    else:
        return Response(detect_hand_and_pen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route("/click")
#@login_required
def click():
    return Response(hand_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')
    #return render_template("analyse.html")
    
@app.route("/distance")
#@login_required
def distance():
    return Response(pose_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')
    #return render_template("analyse.html")


@app.route("/login", methods=['POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_password_correction(attempted_password=form.password.data):
            login_user(attempted_user)
            flash(f'Success! You are logged in as: {attempted_user.username}', category='success')
            return jsonify({'success': True})  # Return success response
        else:
            return jsonify({'success': False, 'message': 'Username and Password do not match'})  # Return failure response
    return jsonify({'success': False, 'message': 'Form validation failed'})  # Return failure response if form validation fails

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            user_to_create = User(
                username=form.username.data,
                email=form.email.data,
                password=form.password1.data
            )
            db.session.add(user_to_create)
            db.session.commit()

            login_user(user_to_create)

            flash(f"Account created, you are now logged in as {user_to_create.username}", category="success")
            return redirect(url_for('login'))

    return render_template("register.html", form=form)

@app.route('/logout')
def logout_page():
    logout_user()
    flash("You have been logged out", category='info')
    return redirect(url_for('index.html'))

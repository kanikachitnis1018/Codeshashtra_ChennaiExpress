from flask import render_template
from flask_login import login_user
from package.forms import LoginForm, RegisterForm
from package.models import User
from package import app, db
import pickle
import numpy as np
import cv2

# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh
# mp_pose = mp.solutions.pose
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/guide")
def guide():
    return render_template("guide.html")

@app.route("/analyse")
def analyse():
    def generate_frames():
        cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Serialize the frame using pickle
            serialized_frame = pickle.dumps(frame)
            
            yield serialized_frame

        cap.release()

    # Create a generator object
    frame_generator = generate_frames()

    # Iterate over the frames and save them into a pickle file
    with open('video_frames.pickle', 'wb') as f:
        for frame in frame_generator:
            pickle.dump(frame, f)
    return render_template("analyse.html")

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
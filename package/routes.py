from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, login_user, logout_user
from package.forms import LoginForm, RegisterForm
from package.models import User
from package import app, db
import matplotlib.pyplot as plt

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
    return Response(detect_hand_and_pen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    #return render_template("analyse.html")
    
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

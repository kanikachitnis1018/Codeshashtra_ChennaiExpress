from flask import render_template
from flask_login import login_user
from package.forms import LoginForm, RegisterForm
from package.models import User
from package import app, db

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/guide")
def guide():
    return render_template("guide.html")

@app.route("/analyse")
def analyse():
    return render_template("analyse.html")

@app.route("/login", methods=['GET','POST'])
def login():
    form = LoginForm()
    return render_template("login.html")

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

    return render_template("register.html")

from flask import Flask, render_template, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SECRET_KEY'] = '6cf1ec880cfcc7bacbe8eb02'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
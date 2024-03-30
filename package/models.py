from package import db, login_manager
from flask_login import UserMixin


@login_manager.user_loader
def load_user(user_id):
    # Load a user object from the database based on the user ID
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(length=20), nullable=False, unique=True)
    email = db.Column(db.String(length=50), nullable=False, unique=True)
    #password_hash = db.Column(db.String(length=60), nullable=False)

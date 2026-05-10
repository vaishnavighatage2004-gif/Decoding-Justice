import hashlib
from utils.db import conn, cursor


def make_hash(password):
    return hashlib.sha256(
        str.encode(password)
    ).hexdigest()


def check_password(password, hashed_text):
    return make_hash(password) == hashed_text


def register_user(username, password):

    hashed_password = make_hash(password)

    cursor.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, hashed_password)
    )

    conn.commit()


def login_user(username, password):

    hashed_password = make_hash(password)

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hashed_password)
    )

    return cursor.fetchone()
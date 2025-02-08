import hashlib

# User data storage
user_data = {
    "admin": hashlib.sha256("admin".encode()).hexdigest()
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    if username in user_data and user_data[username] == hash_password(password):
        return True
    return False

def add_user(username, password):
    if username not in user_data:
        user_data[username] = hash_password(password)
        return True
    return False

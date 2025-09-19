"""
This module implements a Flask API for fake profile detection.
"""
import sqlite3
import re
import bcrypt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from model_service import ModelService
from google.oauth2 import id_token
from google.auth.transport import requests

app = Flask(__name__, static_folder='../frontend')
CORS(app, resources={r"/*": {"origins": "*"}})
model_service = ModelService()

DATABASE = "users.db"
CLIENT_ID = "978383792-ng550h8fkn7aekf2r0uqjn362q0621gp.apps.googleusercontent.com"

def init_db():
    with app.app_context():
        db = sqlite3.connect(DATABASE)
        cursor = db.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, email TEXT UNIQUE, password TEXT)"
        )
        db.commit()

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username")
    email = request.form.get("email")
    password = request.form.get("password")

    if not all([username, email, password]):
        return jsonify({"error": "All fields are required"}), 400

    if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
        return jsonify({"error": "Invalid email format"}), 400

    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    try:
        db = sqlite3.connect(DATABASE)
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed_password),
        )
        db.commit()
        return jsonify({"message": "User registered successfully"})
    except sqlite3.IntegrityError:
        db.rollback()
        return jsonify({"error": "Username or email already exists"}), 409
    finally:
        db.close()

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    if not all([username, password]):
        return jsonify({"error": "All fields are required"}), 400

    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if user and bcrypt.checkpw(password.encode("utf-8"), user[0]):
        return jsonify({"message": "Login successful"})
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route('/google-login', methods=['POST'])
def google_login():
    try:
        # Get token from request - handle both form and JSON data
        token = None
        if request.is_json:
            data = request.get_json()
            token = data.get('token') if data else None
        else:
            token = request.form.get('token')
        
        if not token:
            print("ERROR: No token provided in request")
            return jsonify({"error": "No token provided"}), 400
        
        print(f"DEBUG: Received token (first 50 chars): {token[:50]}...")
        print(f"DEBUG: Token length: {len(token)}")
        print(f"DEBUG: Using CLIENT_ID: {CLIENT_ID}")
        
        # Verify the ID token while checking the CLIENT_ID
        try:
            idinfo = id_token.verify_oauth2_token(
                token, 
                requests.Request(), 
                CLIENT_ID,
                clock_skew_in_seconds=10  # Allow for small time differences
            )
            print(f"DEBUG: Token verification successful")
            print(f"DEBUG: ID info: {idinfo}")
            
        except ValueError as ve:
            print(f"ERROR: Token verification failed: {ve}")
            # More specific error handling
            error_str = str(ve).lower()
            if 'expired' in error_str:
                return jsonify({"error": "Google token has expired. Please sign in again."}), 401
            elif 'audience' in error_str or 'client_id' in error_str:
                return jsonify({"error": "Invalid client configuration"}), 401
            elif 'signature' in error_str:
                return jsonify({"error": "Invalid token signature"}), 401
            else:
                return jsonify({"error": f"Token validation failed: {ve}"}), 401

        # Extract user information
        userid = idinfo.get('sub')
        email = idinfo.get('email')
        name = idinfo.get('name', email.split('@')[0] if email else 'Google User')
        email_verified = idinfo.get('email_verified', False)
        
        if not email:
            print("ERROR: No email found in token")
            return jsonify({"error": "No email found in Google account"}), 400
            
        if not email_verified:
            print("WARNING: Email not verified by Google")
            # You might want to handle unverified emails differently
        
        print(f"DEBUG: User info - ID: {userid}, Email: {email}, Name: {name}")

        # Database operations
        db = sqlite3.connect(DATABASE)
        cursor = db.cursor()

        try:
            # Check if user already exists
            cursor.execute("SELECT id, username, email FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()

            if not user:
                # If user doesn't exist, create a new one
                print(f"DEBUG: Creating new user for email: {email}")
                cursor.execute(
                    "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                    (name, email, "google_user"),
                )
                db.commit()
                print(f"DEBUG: New Google user created successfully")
                user_id = cursor.lastrowid
            else:
                print(f"DEBUG: Existing user found: {user}")
                user_id = user[0]

            return jsonify({
                "message": "Google login successful",
                "user": {
                    "id": user_id,
                    "email": email,
                    "name": name
                }
            })

        except sqlite3.IntegrityError as ie:
            db.rollback()
            print(f"ERROR: Database integrity error: {ie}")
            return jsonify({"error": "Database error: User data conflict"}), 500
        except sqlite3.Error as db_error:
            db.rollback()
            print(f"ERROR: Database error: {db_error}")
            return jsonify({"error": "Database error during user creation"}), 500
        finally:
            db.close()

    except requests.exceptions.RequestException as re:
        print(f"ERROR: Network error during token verification: {re}")
        return jsonify({"error": "Network error during authentication"}), 503
    except Exception as e:
        print(f"ERROR: Unexpected error during Google login: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error during authentication"}), 500

# Add this function to test your Google OAuth setup
@app.route('/test-google-config', methods=['GET'])
def test_google_config():
    """Test endpoint to verify Google OAuth configuration"""
    return jsonify({
        "client_id": CLIENT_ID,
        "client_id_length": len(CLIENT_ID),
        "status": "Google OAuth configuration loaded"
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts if a Twitter profile is fake or real based on the provided data.
    """
    screen_name = request.form.get("screen_name")
    description = request.form.get("description", "")
    followers_count = int(request.form.get("followers_count", 0))
    friends_count = int(request.form.get("friends_count", 0))
    statuses_count = int(request.form.get("statuses_count", 0))

    if not screen_name:
        return jsonify({"error": "Screen name is required"}), 400

    # Input validation for unrealistic values
    if followers_count > 100000000000 or friends_count > 100000000000 or statuses_count > 100000000000:
        return jsonify({"prediction": "fake", "confidence": 100.0})

    # Heuristic for random-looking text based on vowel ratio
    def has_low_vowel_ratio(text):
        if not isinstance(text, str) or not text or ' ' in text:
            return False
        vowels = "aeiouAEIOU"
        num_vowels = sum(1 for char in text if char in vowels)
        ratio = num_vowels / len(text) if len(text) > 0 else 0
        # Flag as suspicious if length is > 5 and vowel ratio is low
        return len(text) > 5 and ratio < 0.2

    if has_low_vowel_ratio(screen_name) or has_low_vowel_ratio(description):
        return jsonify({"prediction": "fake", "confidence": {"real": 0.0, "fake": 100.0}})

    data = {
        "description": description,
        "followers_count": followers_count,
        "friends_count": friends_count,
        "statuses_count": statuses_count,
    }

    result = model_service.predict(data)
    return jsonify(result)

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
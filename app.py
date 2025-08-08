# app.py
from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

# Mood to playlist mapping
playlists = {
    "happy": ["Happy Song 1", "Happy Song 2", "Happy Song 3"],
    "sad": ["Sad Song 1", "Sad Song 2", "Sad Song 3"],
    "relaxed": ["Relax Song 1", "Relax Song 2", "Relax Song 3"],
    "energetic": ["Energy Song 1", "Energy Song 2", "Energy Song 3"]
}

# Initialize app
app = Flask(__name__)
CORS(app)

# Home route for testing
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Backend is running ✅",
        "message": "Send a POST request to /predict with {'text': 'your sentence'}"
    })

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        user_text = data.get("text", "")

        if not user_text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Transform and predict mood
        X = vectorizer.transform([user_text])
        mood = model.predict(X)[0]

        return jsonify({
            "mood": mood,
            "playlist": playlists.get(mood, [])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # host='0.0.0.0' is important for Render to make the app externally accessible
    app.run(host="0.0.0.0", port=5000, debug=True)

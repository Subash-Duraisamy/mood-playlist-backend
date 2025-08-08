# app.py
from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

# Load model
with open('model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

# Mood to playlist mapping
playlists = {
    "happy": ["Happy Song 1", "Happy Song 2", "Happy Song 3"],
    "sad": ["Sad Song 1", "Sad Song 2", "Sad Song 3"],
    "relaxed": ["Relax Song 1", "Relax Song 2", "Relax Song 3"],
    "energetic": ["Energy Song 1", "Energy Song 2", "Energy Song 3"]
}

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_text = data.get("text", "")

    X = vectorizer.transform([user_text])
    mood = model.predict(X)[0]

    return jsonify({
        "mood": mood,
        "playlist": playlists[mood]
    })

if __name__ == "__main__":
    app.run(debug=True)

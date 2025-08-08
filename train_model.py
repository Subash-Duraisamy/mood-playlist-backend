# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample mood dataset
data = {
    'text': [
        "I am feeling so happy and excited",
        "Life is beautiful and joyful",
        "Feeling sad and down today",
        "I am depressed and lonely",
        "So relaxed and calm right now",
        "I feel peaceful and at ease",
        "I am full of energy and ready to go",
        "Feeling tired but still motivated"
    ],
    'mood': [
        "happy", "happy",
        "sad", "sad",
        "relaxed", "relaxed",
        "energetic", "energetic"
    ]
}

df = pd.DataFrame(data)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['mood']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model & vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("Model trained and saved as model.pkl")

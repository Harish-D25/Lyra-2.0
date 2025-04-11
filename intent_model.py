from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample data
samples = [
    ("what's your name", "get_name"),
    ("who are you", "get_name"),
    ("how are you", "get_feeling"),
    ("what can you do", "capabilities"),
    ("tell me a joke", "joke"),
    ("exit", "exit"),
    ("bye", "exit"),
    ("what is the weather like today", "weather"),
]

# Split into texts and labels
X = [x[0] for x in samples]
y = [x[1] for x in samples]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Classifier
clf = LogisticRegression()
clf.fit(X_vec, y)

# Save model and vectorizer
joblib.dump(clf, "intent_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

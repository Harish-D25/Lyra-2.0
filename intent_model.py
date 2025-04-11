from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample data
samples = [
    # get_name
    ("what's your name", "get_name"),
    ("who are you", "get_name"),
    
    # get_feeling
    ("how are you", "get_feeling"),
    ("how do you feel", "get_feeling"),
    
    # capabilities
    ("what can you do", "capabilities"),
    ("what are your features", "capabilities"),

    # joke
    ("tell me a joke", "joke"),
    ("make me laugh", "joke"),

    # exit
    ("exit", "exit"),
    ("bye", "exit"),
    ("goodbye", "exit"),

    # greeting
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("yo dude", "greeting"),
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

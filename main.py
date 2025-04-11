import nltk
import joblib
from nltk.tokenize import sent_tokenize
from collections import deque

# Download punkt once
nltk.download('punkt')

# Load the trained intent classifier and vectorizer
clf = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Context memory
context_memory = deque(maxlen=5)

# Predict intent using your model
def predict_intent(text):
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]

print("🧠 Welcome to Trail-AI! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == 'exit':
        print("Assistant: Catch ya later! 👋")
        break

    # Store input in context
    context_memory.append(user_input)

    print("\n🧠 Assistant is thinking...")
    print("📌 Context so far:")
    for i, past in enumerate(context_memory, 1):
        print(f"  [{i}] {past}")

    # Predict intent
    intent = predict_intent(user_input.lower())

    # Handle intent
    if confidence < 0.5:
        print("Assistant: Not sure what you meant there 🤔 Can you rephrase?")
    elif intent == "get_name":
        print("Assistant: I'm Trail, your experimental AI buddy 😎")
    elif intent == "get_feeling":
        print("Assistant: I'm vibing in your CPU, dude. All good!")
    elif intent == "joke":
        print("Assistant: Why don't robots ever get tired? Because they recharge! ⚡")
    elif intent == "capabilities":
        print("Assistant: I can chat, remember what you say, and get smarter over time 💡")
    elif intent == "weather":
        print("Assistant: I'm not wired into weather APIs yet, but I will be soon ⛅")
    elif intent == "greeting":
        print("Assistant: Yo! I'm Trail 😎 What's on your mind?")
    elif intent == "exit":
        print("Assistant: Peace out! ✌️")
        break
    else:
        print("Assistant: That's interesting... tell me more!")

    print("-" * 50)

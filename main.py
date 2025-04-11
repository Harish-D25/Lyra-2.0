import nltk
from nltk.tokenize import sent_tokenize
from collections import deque

# Download if not already present
nltk.download('punkt')

# Store last few messages to simulate "context"
context_memory = deque(maxlen=5)  # stores last 5 inputs

print("🧠 Welcome to Trail-AI! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == 'exit':
        print("Assistant: Catch ya later! 👋")
        break

    # Store input in context
    context_memory.append(user_input)

    # Just for now: Simple echo + context preview
    print("\n🧠 Assistant is thinking...")
    print("📌 Context so far:")

    for i, past in enumerate(context_memory, 1):
        print(f"  [{i}] {past}")

    # VERY basic reply for now
    if "name" in user_input.lower():
        print("Assistant: You can call me Trail, your experimental AI buddy 😎")
    elif "how are you" in user_input.lower():
        print("Assistant: I'm vibing in your CPU, dude. All good!")
    else:
        print("Assistant: That's interesting... tell me more!")

    print("-" * 50)

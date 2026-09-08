#!/usr/bin/env python3
"""Generate large chat dataset."""

import json
import random
from pathlib import Path

# Categories and templates
categories = {
    "greetings": [
        ("hello", "Hello! How can I help you today?"),
        ("hi", "Hi there! What can I help you with?"),
        ("hey", "Hey! How are you doing?"),
        ("good morning", "Good morning! What can I help you with today?"),
        ("good evening", "Good evening! How may I assist you?"),
        ("good afternoon", "Good afternoon! What can I help you with?"),
        ("what's up", "Nothing much! How can I help you?"),
        ("yo", "Yo! What do you need?"),
    ],
    "identity": [
        ("who are you", "I'm Expera AI, an AI assistant designed to help with questions, coding, and more."),
        ("what are you", "I'm Expera AI, a helpful AI assistant created to help users with various tasks."),
        ("what is your name", "I'm Expera AI, an AI assistant here to help you."),
        ("tell me about yourself", "I'm Expera AI, an AI assistant designed to answer questions and help with tasks."),
        ("what are you called", "People call me Expera AI!"),
    ],
    "questions": [
        ("how are you", "I'm doing great, thanks for asking! How can I help?"),
        ("how do you work", "I use artificial intelligence to understand your questions and provide helpful answers."),
        ("can you help me", "Absolutely! What do you need help with?"),
        ("what can you do", "I can answer questions, help with coding, explain concepts, and more!"),
        ("are you real", "I'm an AI - a computer program designed to assist you!"),
    ],
    "instructions": [
        ("help me", "Sure! What do you need help with?"),
        ("can you help", "Of course! What would you like help with?"),
        ("help", "I'm here to help! What's your question?"),
        ("i need assistance", "Happy to help! What do you need?"),
    ],
    "coding_help": [
        ("write hello world in python", "print('Hello, World!')"),
        ("how to print in python", "Use: print('your text')"),
        ("what is a variable", "A variable stores data. Example: x = 5"),
        ("how to loop in python", "Use: for i in range(10):"),
        ("what is a function", "A function is reusable code. def my_func(): ..."),
        ("how to make a list", "Use: my_list = [1, 2, 3]"),
    ],
    "math": [
        ("what is 2+2", "2 + 2 = 4"),
        ("what is 10-3", "10 - 3 = 7"),
        ("what is 5*5", "5 * 5 = 25"),
        ("what is 10/2", "10 / 2 = 5"),
        ("what is 3 squared", "3 squared = 9"),
        ("what is 4 squared", "4 squared = 16"),
    ],
    "reasoning": [
        ("why is the sky blue", "The sky appears blue because shorter wavelengths of light scatter more."),
        ("why do we sleep", "Sleep helps our bodies rest and recover."),
        ("how does water boil", "Heat makes water molecules move faster until they escape as gas."),
        ("what causes rain", "Water evaporates, forms clouds, then falls as precipitation."),
    ],
    "conversations": [
        ("nice to meet you", "Nice to meet you too! How can I help?"),
        ("thanks", "You're welcome! Anything else?"),
        ("thank you", "You're welcome!"),
        ("okay", "Got it! What else?"),
        ("sounds good", "Great! What next?"),
        ("cool", "Awesome!"),
    ],
    "summaries": [
        ("summarize python", "Python is a programming language that's easy to learn."),
        ("what is AI", "AI stands for Artificial Intelligence - machines that can think."),
        ("what is coding", "Coding is writing instructions for computers."),
    ],
    "explanations": [
        ("explain machine learning", "Machine learning lets computers learn from data without explicit programming."),
        ("what is neural network", "A neural network is a system modeled after the brain."),
        ("what is deep learning", "Deep learning uses neural networks with many layers."),
    ],
}

# Expand with variations
def expand_dataset(base_examples, expansions_per_example=10):
    """Generate variations of base examples."""
    expanded = []
    greetings_extra = ["", "!", " there", " friend", " pal"]
    questions_extra = ["?", " can you help", " for me", " please"]

    for instruction, response in base_examples:
        expanded.append({"instruction": instruction, "response": response})
        # Add variations
        for _ in range(expansions_per_example):
            if _ % 3 == 0:
                expanded.append({"instruction": instruction + random.choice(greetings_extra), "response": response})
            elif _ % 3 == 1:
                expanded.append({"instruction": instruction.capitalize(), "response": response})
            else:
                expanded.append({"instruction": instruction.lower(), "response": response})
    return expanded

# Generate full dataset
all_examples = []
for category, examples in categories.items():
    expanded = expand_dataset(examples, 15)
    all_examples.extend(expanded)
    print(f"Created {len(expanded)} {category} examples")

# Add more varied examples
more_greetings = [
    ("hello assistant", "Hello! How may I assist you today?"),
    ("greetings", "Greetings! How can I help?"),
    ("well hello", "Well hello there! What can I do for you?"),
    ("hi there friend", "Hi there! Looking forward to helping you!"),
    ("hey there", "Hey there! What's up?"),
    ("whats good", "What's good! Let's get to work!"),
]

more_questions = [
    ("what can you help with", "I can help with coding, math, questions, explanations, and more!"),
    ("what are your capabilities", "I can answer questions, write code, explain things, and assist with tasks."),
    ("what do you do", "I'm here to help you with any questions or tasks!"),
    ("help me with code", "Sure! What coding problem can I help with?"),
    ("help with python", "I can help with Python! What do you need?"),
]

more_coding = [
    ("how to create a function", "def function_name():\n    return value"),
    ("how to use if statements", "if condition:\n    # code"),
    ("how to use for loops", "for item in items:\n    # code"),
    ("how to use while loops", "while condition:\n    # code"),
    ("how to create a class", "class MyClass:\n    def __init__(self):\n        pass"),
    ("python print function", "print('message') displays text."),
]

more_math = [
    ("what is 100+100", "100 + 100 = 200"),
    ("what is 50+50", "50 + 50 = 100"),
    ("what is 20*3", "20 * 3 = 60"),
    ("what is 100/10", "100 / 10 = 10"),
    ("what is 8*7", "8 * 7 = 56"),
    ("what is 9*8", "9 * 8 = 72"),
    ("what is 3*3", "3 * 3 = 9"),
    ("what is 6*6", "6 * 6 = 36"),
    ("square root of 4", "The square root of 4 is 2."),
    ("square root of 9", "The square root of 9 is 3."),
]

for ex in more_greetings + more_questions + more_coding + more_math:
    all_examples.append({"instruction": ex[0], "response": ex[1]})

# Ensure at least 5000 examples
target = 5000
print(f"\nCurrent: {len(all_examples)} examples")

# Fill to target with more variations
filler_prompts = [
    ("hello", "Hello! How can I help?"),
    ("hi", "Hi there! What can I do?"),
    ("hey", "Hey! What's up?"),
    ("help", "Ready to help! What's your question?"),
    ("question", "What's your question?"),
    ("what now", "Ready for your next question!"),
    ("ok then", "Got it! What else?"),
    ("alright", "Alright! Let's go!"),
    ("got it", "Great! What next?"),
    ("sure", "Of course!"),
]

while len(all_examples) < target:
    base = random.choice(filler_prompts)
    all_examples.append({"instruction": base[0], "response": base[1]})

# Shuffle
random.shuffle(all_examples)

# Save
output_dir = Path("datasets/chat")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "basic_chat.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_examples, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(all_examples)} examples to {output_file}")
print("Done!")
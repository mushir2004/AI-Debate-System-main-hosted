from ai_engine import generate_rebuttal, detect_fallacies, steelman_argument

topic = "Should AI be used in courtrooms?"
user_input = "If we let AI into courtrooms, it will just blindly lock everyone up because computers have no empathy! It's a terrible idea."

print("--- 1. Testing Rebuttal (Pro Stance) ---")
print(generate_rebuttal(topic, "PRO AI in courtrooms", user_input))

print("\n--- 2. Testing Fallacy Detection ---")
print(detect_fallacies(user_input))

print("\n--- 3. Testing Steelmanning ---")
print(steelman_argument(user_input))
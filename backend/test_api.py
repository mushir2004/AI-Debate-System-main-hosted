import requests

url = "http://localhost:11434/api/generate"

data = {
    # Make sure this matches the exact name from the 'ollama list' command
    "model": "llama3", 
    "prompt": "Explain the concept of 'steelmanning' an argument in one short paragraph.",
    "stream": False 
}

print("Sending request to local AI...")

response = requests.post(url, json=data)

# Better error handling
if response.status_code == 200:
    result = response.json()
    print("\n--- AI Response ---")
    print(result["response"])
else:
    print(f"\nFailed to connect. Status Code: {response.status_code}")
    print(f"Error Details from Ollama: {response.text}")
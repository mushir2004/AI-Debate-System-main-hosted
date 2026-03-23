import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load the hidden API key from the .env file
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError("NVIDIA_API_KEY is missing! Check your .env file.")

# Initialize the client with the NVIDIA API endpoint
client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=api_key
)

# Use the massive 70B model available on NVIDIA NIM
MODEL = "meta/llama-3.3-70b-instruct" 

def query_model(messages, require_json=False):
    """Helper function to communicate with the NVIDIA API via the OpenAI SDK."""
    try:
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.3,
        }
        
        # Enforce JSON formatting if requested
        if require_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error connecting to NVIDIA AI: {e}")
        # Return a safe fallback so the backend doesn't crash
        if require_json:
            return '{"error": "API Connection Failed"}'
        return f"Error connecting to AI: {e}"

def generate_rebuttal(topic, stance, user_argument):
    """Generates a counter-argument based on the assigned stance."""
    system_prompt = f"You are a highly competitive, logical debater. The topic is: '{topic}'. Your strict stance is: {stance}. Provide a concise, forceful, and logical 1-paragraph counter-argument to the user's latest point. Do not be overly polite."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_argument}
    ]
    return query_model(messages)

def detect_fallacies(user_argument):
    """Analyzes the user's argument for logical fallacies and returns JSON."""
    system_prompt = """Analyze the user's argument for logical fallacies. 
    Use ONLY this taxonomy: Straw Man, Ad Hominem, False Dichotomy, Slippery Slope, Red Herring, or None.
    Respond ONLY in strict JSON format like this:
    {"fallacy": "Name of Fallacy or None", "explanation": "One short sentence explaining why."}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_argument}
    ]
    
    raw_response = query_model(messages, require_json=True)
    
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {"fallacy": "Error", "explanation": "Failed to parse AI response into JSON."}

def steelman_argument(user_argument):
    """Reconstructs the user's argument into its strongest, most logical form."""
    system_prompt = "You are an expert logician. Take the user's argument, strip away any emotional language, hyperbole, or weak logic, and reconstruct it into its most bulletproof, rigorous, and charitable form. Output ONLY the strengthened argument."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_argument}
    ]
    return query_model(messages)

def judge_debate(transcript_text):
    """Reads the debate history, picks a winner, and generates a brag quote."""
    system_prompt = """You are an impartial, expert debate judge. Read the following debate transcript.
    Decide who won: "Pro" or "Con". 
    Then, write a 1-sentence reason why they won.
    Finally, write a 1-sentence 'brag' quote from the winner's perspective (e.g., "I absolutely dismantled their argument with pure logic!").
    
    Respond ONLY in strict JSON format like this:
    {"winner": "Pro or Con", "reason": "They had better evidence.", "brag": "I crushed their silly argument!"}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript_text}
    ]
    
    raw_response = query_model(messages, require_json=True)
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {"winner": "Tie", "reason": "Too close to call.", "brag": "It was a legendary battle of wits."}
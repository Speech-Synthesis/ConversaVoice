import os
import sys
from dotenv import load_dotenv

# Load env vars
load_dotenv()

try:
    from groq import Groq
    print("Groq library imported successfully.")
except ImportError as e:
    print(f"Failed to import groq: {e}")
    sys.exit(1)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in environment.")
    sys.exit(1)

print(f"API Key present: {api_key[:5]}...{api_key[-5:]}")

try:
    print("Initializing Groq client...")
    client = Groq(api_key=api_key)
    
    print("Sending test request...")
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, say test."}
        ],
        temperature=0.7,
        max_tokens=10
    )
    
    print("Response received:")
    print(completion.choices[0].message.content)

except Exception as e:
    print(f"\n[ERROR] Groq API call failed: {e}")

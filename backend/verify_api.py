"""Verification script for API endpoints."""

import os
import sys
import time
import argparse
import requests
import uuid
from dotenv import load_dotenv

# Load env
load_dotenv()

BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

def print_status(component, status, message=""):
    color = "\033[92m" if status == "PASS" else "\033[91m"
    reset = "\033[0m"
    print(f"[{component}] {color}{status}{reset} {message}")

def test_health():
    """Test health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print_status("Health", "PASS", f"Status: {response.json()['status']}")
            return True
        else:
            print_status("Health", "FAIL", f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_status("Health", "FAIL", f"Error: {e}")
        return False

def test_session():
    """Test session creation."""
    try:
        response = requests.post(f"{BASE_URL}/api/session", timeout=5)
        if response.status_code == 200:
            session_id = response.json().get("session_id")
            if session_id:
                print_status("Session", "PASS", f"Created session: {session_id}")
                return session_id
            else:
                print_status("Session", "FAIL", "No session ID returned")
        else:
            print_status("Session", "FAIL", f"Status code: {response.status_code}")
    except Exception as e:
        print_status("Session", "FAIL", f"Error: {e}")
    return None

def test_chat(session_id):
    """Test chat endpoint."""
    try:
        payload = {
            "text": "Hello, this is a test.",
            "session_id": session_id
        }
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=30)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get("response")
            style = data.get("style", "neutral")
            print_status("Chat", "PASS", f"Response: '{response_text[:30]}...' (Style: {style}) - Time: {duration:.2f}s")
            return response_text, style
        else:
            print_status("Chat", "FAIL", f"Status code: {response.status_code} - {response.text}")
    except Exception as e:
        print_status("Chat", "FAIL", f"Error: {e}")
    return None, None

def test_synthesize(text, style):
    """Test synthesis endpoint."""
    if not text:
        print_status("TTS", "SKIP", "No text to synthesize")
        return
        
    try:
        payload = {
            "text": text,
            "style": style
        }
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/synthesize", json=payload, timeout=60)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            audio_url = data.get("audio_url")
            print_status("TTS", "PASS", f"Audio URL: {audio_url} - Time: {duration:.2f}s")
            
            # Test audio download
            if audio_url:
                try:
                    full_url = f"{BASE_URL}{audio_url}"
                    audio_resp = requests.get(full_url, timeout=10)
                    if audio_resp.status_code == 200:
                        size = len(audio_resp.content)
                        print_status("Audio", "PASS", f"Downloaded {size} bytes")
                    else:
                        print_status("Audio", "FAIL", f"Download failed: {audio_resp.status_code}")
                except Exception as e:
                    print_status("Audio", "FAIL", f"Download error: {e}")
        else:
            print_status("TTS", "FAIL", f"Status code: {response.status_code} - {response.text}")
    except Exception as e:
        print_status("TTS", "FAIL", f"Error: {e}")

def main():
    print(f"Starting API Verification against {BASE_URL}...\n")
    
    # 1. Check Health
    if test_health():
        # 2. Create Session
        session_id = test_session()
        
        if session_id:
            # 3. Test Chat
            response_text, style = test_chat(session_id)
            
            # 4. Test TTS
            test_synthesize(response_text, style)
            
            print("\nVerification Complete.")
        else:
            print("\nSkipping remaining tests due to session failure.")
    else:
        print("\nSkipping remaining tests due to health check failure.")

if __name__ == "__main__":
    try:
        # Enable colored output on Windows
        os.system("")
    except:
        pass
    main()

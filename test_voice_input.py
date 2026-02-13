"""
Quick test to verify Groq Whisper voice input works.
Run this to test microphone → Groq Whisper transcription.
"""

import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_voice_input():
    """Test voice input with Groq Whisper."""
    print("=" * 60)
    print("Testing Voice Input with Groq Whisper")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found!")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    # Check STT backend
    stt_backend = os.getenv("STT_BACKEND", "groq")
    print(f"✅ STT_BACKEND: {stt_backend}")
    
    try:
        from src.orchestrator import Orchestrator
        
        print("\n🔧 Initializing orchestrator...")
        orch = Orchestrator()
        await orch.initialize()
        print("✅ Orchestrator initialized")
        
        print("\n🎤 Initializing STT...")
        await orch.initialize_stt()
        print(f"✅ STT initialized: {type(orch._stt_client).__name__}")
        
        if type(orch._stt_client).__name__ != "GroqWhisperClient":
            print(f"⚠️  Warning: Using {type(orch._stt_client).__name__} instead of GroqWhisperClient")
            print("   Make sure STT_BACKEND=groq in .env")
        
        print("\n" + "=" * 60)
        print("🎙️  SPEAK NOW! (You have 10 seconds)")
        print("=" * 60)
        print("Say something like: 'Hello, can you hear me?'")
        print()
        
        # Use listen_once from the STT client directly
        text = orch._stt_client.listen_once(timeout=10.0)
        
        if text:
            print("\n" + "=" * 60)
            print(f"✅ Transcribed: \"{text}\"")
            print("=" * 60)
            print("\n🎉 Voice input works with Groq Whisper!")
            return True
        else:
            print("\n⚠️  No speech detected")
            print("   Possible issues:")
            print("   - Microphone not working")
            print("   - Didn't speak loud enough")
            print("   - Background noise too high")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'orch' in locals():
            await orch.shutdown()

if __name__ == "__main__":
    success = asyncio.run(test_voice_input())
    
    if success:
        print("\n✅ Groq Whisper voice input is working!")
        print("   The issue might be with the Streamlit frontend.")
    else:
        print("\n❌ Voice input test failed")
        print("   Check your microphone and API key")

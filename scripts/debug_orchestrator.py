import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.orchestrator import Orchestrator

async def diagnostic():
    logging.basicConfig(level=logging.INFO)
    print("Starting diagnostic...")
    
    try:
        orchestrator = Orchestrator(session_id="test-diag")
        print("Initializing orchestrator...")
        await orchestrator.initialize()
        print("Initialization successful.")
        
        print("Processing text 'hello'...")
        result = await orchestrator.process_text("hello", speak=False)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"\n!!! ERROR CAUGHT !!!")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnostic())

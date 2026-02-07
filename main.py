#!/usr/bin/env python3
"""
ConversaVoice - Context-aware voice assistant with emotional intelligence.

Entry point for the full assistant pipeline.
"""

import asyncio
import argparse
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ConversaVoice - Emotionally intelligent voice assistant"
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session ID for conversation persistence (auto-generated if not provided)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Process a single text input and exit"
    )
    parser.add_argument(
        "--voice", "-v",
        action="store_true",
        help="Use voice input mode (microphone with Whisper STT)"
    )

    args = parser.parse_args()

    # Import here to allow --help without loading all dependencies
    from src.orchestrator import Orchestrator

    # Create orchestrator
    orchestrator = Orchestrator(session_id=args.session)

    if args.text:
        # Single text processing mode
        async def process_single():
            await orchestrator.initialize()
            result = await orchestrator.process_text(args.text)
            print(f"\nResponse: {result.assistant_response}")
            print(f"Style: {result.style}")
            print(f"Latency: {result.latency_ms:.0f}ms")
            await orchestrator.shutdown()

        asyncio.run(process_single())
    else:
        # Interactive mode (text or voice)
        asyncio.run(orchestrator.run_interactive(use_voice=args.voice))


if __name__ == "__main__":
    main()

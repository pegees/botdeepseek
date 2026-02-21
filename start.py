#!/usr/bin/env python3
"""
🚀 ONE-CLICK START
==================
Run this to start the Scalp Signal Bot.

Usage:
    python start.py
"""
import os
import sys
import subprocess
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)

def check_env():
    """Check if API_KEYS.env or .env file exists."""
    api_keys_file = SCRIPT_DIR / "API_KEYS.env"
    env_file = SCRIPT_DIR / ".env"

    # Prefer API_KEYS.env if it exists
    if api_keys_file.exists():
        return "API_KEYS.env"
    elif env_file.exists():
        return ".env"
    else:
        print("❌ No API keys file found!")
        print("   Create API_KEYS.env and add your keys.")
        return None

def check_deps():
    """Check if dependencies are installed."""
    try:
        import telegram
        import aiohttp
        import numpy
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e.name}")
        print("   Run: pip install -r requirements.txt")
        return False

def main():
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 SCALP SIGNAL BOT - ONE CLICK START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

    # Check environment
    env_file = check_env()
    if not env_file:
        sys.exit(1)

    # Load API keys
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR / env_file)
    print(f"✅ Loaded keys from {env_file}")

    # Quick validation
    token = os.getenv("TELEGRAM_TOKEN")
    if not token or token == "your_bot_token_from_botfather":
        print("❌ TELEGRAM_TOKEN not configured in .env")
        sys.exit(1)

    print("✅ Starting bot...")
    print()

    # Import and run
    try:
        from main import main as run_bot
        run_bot()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

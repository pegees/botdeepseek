#!/bin/bash
# 🚀 ONE-CLICK START - Just run: ./run.sh

cd "$(dirname "$0")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 SCALP SIGNAL BOT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check for .env
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Copy .env.example to .env and fill in your keys"
    exit 1
fi

# Check for venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install deps if needed
pip3 install -q -r requirements.txt 2>/dev/null

echo "✅ Starting bot..."
python3 start.py

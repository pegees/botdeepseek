#!/bin/bash
# Double-click this file to start the Telegram bot

cd "$(dirname "$0")"

echo "=================================================="
echo "  TELEGRAM SCALP SCANNER BOT"
echo "  Control everything from Telegram!"
echo "=================================================="
echo ""
echo "Commands available in Telegram:"
echo "  /scan       - Full scan (top 30 pairs)"
echo "  /scan BTC   - Deep scan single pair"
echo "  /scan meme  - Volatile pairs only"
echo "  /top        - Quick top 10 movers"
echo "  /status     - Bot health check"
echo "  /autoscan   - Auto-scan every 15 min"
echo ""
echo "Starting bot..."
echo ""

# Activate venv if exists
[ -d "venv" ] && source venv/bin/activate

# Run the Telegram bot
python3 telegram_bot.py

# Keep window open if error
echo ""
read -p "Press Enter to close..."

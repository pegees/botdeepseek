# Scalp Signal Bot

A Telegram bot that scans **ALL** Binance Futures pairs (200+) and uses DeepSeek AI to find the best scalp trading setups.

## Quick Start

### Double-click to start:
**Mac:** Double-click `START.command`

A browser window will open where you can:
1. Enter your Telegram Bot Token
2. Enter your DeepSeek API Key
3. Click "Start Bot"

That's it!

![Launcher Screenshot](https://via.placeholder.com/400x300?text=Web+Launcher)

## Get Your API Keys

1. **Telegram Bot Token**
   - Open Telegram and message [@BotFather](https://t.me/BotFather)
   - Send `/newbot` and follow instructions
   - Copy the token it gives you

2. **DeepSeek API Key**
   - Go to [platform.deepseek.com](https://platform.deepseek.com/)
   - Sign up and create an API key
   - Copy the key

## Features

- **Scans ALL pairs** - 200+ USDT perpetuals, not just top coins
- **AI-powered** - DeepSeek analyzes market data for best setups
- **Signal history** - Track past signals with `/history`
- **Scheduled alerts** - Enable with `/alert on`
- **Natural language** - Just say "find me a long" or "any shorts on 5m?"

## Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/help` | Full command list |
| `/scalp` | Get best scalp signal |
| `/scalp 5m` | Signal on 5m timeframe |
| `/multi` | Get top 3 setups |
| `/history` | Your recent signals |
| `/status` | Bot health + pair count |
| `/pairs` | Show scanned pairs |
| `/alert on` | Enable scheduled alerts |
| `/alert off` | Disable alerts |

## Natural Language

Just type naturally:
- "give me a scalp setup"
- "find me a long on 5m"
- "any good shorts?"
- "top 3 setups"

## Timeframes

`1m` `3m` `5m` `15m` `30m` `1h` `4h`

## Alternative: Command Line

If you prefer terminal:
```bash
python3 launcher.py
```

Or the old way:
```bash
./run.sh
```

## Disclaimer

This bot is for educational purposes. Trading crypto involves risk. Always:
- Do your own research
- Never risk more than you can afford to lose
- Use proper risk management

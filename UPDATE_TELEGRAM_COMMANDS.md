# UPDATE: Telegram-Controlled Scanner (Not CLI)

> **Problem:** The scanner only runs when you execute the Python script manually from terminal. We want to control everything FROM Telegram — type a command, bot scans, bot replies with results.

---

## WHAT WE WANT

The bot should be a Telegram bot that is ALWAYS RUNNING and waiting for commands. When you send a message in Telegram, it does the work and replies back in the same chat.

Commands to support:

| Command | What it does |
|---|---|
| /scan | Full scan — top 50 pairs, analyze all, DeepSeek picks best setups, reply in chat |
| /scan BTC | Scan a specific pair — deep analysis on just that one pair |
| /scan meme | Scan only high-volatility/meme pairs (top movers by % change) |
| /top | Quick list — show top 10 movers right now (pair, price, % change, volume) without DeepSeek |
| /status | Bot health check — is Binance connected, is DeepSeek working, last scan time |
| /autoscan on | Turn on auto-scan every 15 minutes (on candle close) |
| /autoscan off | Turn off auto-scan |

---

## HOW TO RESTRUCTURE THE BOT

The bot needs TWO things running at the same time:
1. Telegram bot listener (always on, waiting for commands)
2. Optional auto-scanner (runs every 15 min if enabled)

### Main entry point — `main.py`

```python
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timezone

# Import your scanner functions
from scanner import (
    get_exchange,
    get_scannable_pairs,
    scan_all_pairs,
    build_deepseek_prompt,
    ask_deepseek,
    analyze_pair,
    clean_for_telegram
)

# ============================================
# CONFIG
# ============================================
TELEGRAM_BOT_TOKEN = "your-bot-token"
TELEGRAM_ADMIN_ID = "your-telegram-user-id"  # Only you can control the bot
DEEPSEEK_API_KEY = "your-deepseek-key"

# ============================================
# SETUP
# ============================================
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
scheduler = AsyncIOScheduler()
exchange = get_exchange()

# State
autoscan_enabled = False
last_scan_time = None
is_scanning = False

# ============================================
# HELPER: Run scan and return result text
# ============================================
async def run_full_scan(mode="all", specific_pair=None):
    """
    Core scan function. Called by both commands and auto-scanner.
    
    mode: "all" = full scan, "meme" = volatile movers only
    specific_pair: if set, only scan this one pair
    """
    global last_scan_time, is_scanning
    
    if is_scanning:
        return "Already scanning, wait for current scan to finish."
    
    is_scanning = True
    
    try:
        if specific_pair:
            # Single pair deep analysis
            symbol = specific_pair.upper()
            if not symbol.endswith("/USDT:USDT"):
                symbol = f"{symbol}/USDT:USDT"
            
            analysis = analyze_pair(exchange, symbol)
            if not analysis:
                return f"Could not analyze {symbol}. Check if the pair exists on Binance Futures."
            
            # Send single pair to DeepSeek
            prompt = build_deepseek_prompt([analysis])
            result = ask_deepseek(prompt)
            pair_count = 1
        
        else:
            # Full market scan
            top_pairs = get_scannable_pairs(exchange, top_n=50)
            
            if mode == "meme":
                # Filter to only high-volatility pairs (>3% daily change)
                top_pairs = [p for p in top_pairs if abs(p.get("change", 0)) > 3]
                if not top_pairs:
                    return "No meme/volatile pairs found with >3% daily change right now."
            
            if not top_pairs:
                return "No pairs found. Binance API might be down."
            
            all_data = scan_all_pairs(exchange, top_pairs)
            
            if not all_data:
                return "Could not analyze any pairs. Check API connection."
            
            prompt = build_deepseek_prompt(all_data)
            result = ask_deepseek(prompt)
            pair_count = len(all_data)
        
        if not result:
            return "DeepSeek returned empty response. Check API key."
        
        last_scan_time = datetime.now(timezone.utc)
        timestamp = last_scan_time.strftime("%Y-%m-%d %H:%M UTC")
        
        header = f"SCALP SCAN — {timestamp}\n"
        header += f"Timeframe: 15m | Pairs scanned: {pair_count}\n"
        if mode == "meme":
            header += "Mode: MEME/HIGH-VOL only\n"
        header += "\n"
        
        return header + clean_for_telegram(result)
    
    finally:
        is_scanning = False


# ============================================
# TELEGRAM COMMANDS
# ============================================

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.reply(
        "Scalp Scanner Bot — active\n\n"
        "Commands:\n"
        "/scan — full market scan (top 50 pairs)\n"
        "/scan BTC — deep scan on specific pair\n"
        "/scan meme — scan only volatile/meme pairs\n"
        "/top — quick top 10 movers\n"
        "/status — bot health check\n"
        "/autoscan on — auto-scan every 15 min\n"
        "/autoscan off — stop auto-scan"
    )


@dp.message(Command("scan"))
async def cmd_scan(message: types.Message, command: CommandObject):
    """
    /scan — full scan
    /scan BTC — scan specific pair
    /scan meme — scan volatile movers only
    """
    # Check if user is admin
    if str(message.from_user.id) != TELEGRAM_ADMIN_ID:
        await message.reply("Not authorized.")
        return
    
    args = command.args  # Everything after /scan
    
    if args and args.strip().lower() == "meme":
        await message.reply("Scanning volatile/meme pairs... ~30 seconds")
        result = await run_full_scan(mode="meme")
    
    elif args and args.strip():
        # Specific pair scan
        pair = args.strip().upper()
        await message.reply(f"Deep scanning {pair}... ~15 seconds")
        result = await run_full_scan(specific_pair=pair)
    
    else:
        # Full scan
        await message.reply("Scanning top 50 pairs... ~60 seconds")
        result = await run_full_scan(mode="all")
    
    # Send result (split if too long)
    await send_long_message(message.chat.id, result)


@dp.message(Command("top"))
async def cmd_top(message: types.Message):
    """Quick top 10 movers — no DeepSeek, just raw data"""
    if str(message.from_user.id) != TELEGRAM_ADMIN_ID:
        return
    
    await message.reply("Fetching top movers...")
    
    try:
        tickers = exchange.fetch_tickers()
        
        perp_pairs = []
        skip = ["USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP"]
        for symbol, t in tickers.items():
            if not symbol.endswith("/USDT:USDT"):
                continue
            base = symbol.split("/")[0]
            if base in skip:
                continue
            vol = t.get("quoteVolume") or 0
            if vol < 100_000:
                continue
            perp_pairs.append({
                "symbol": base,
                "price": t.get("last", 0),
                "change": t.get("percentage", 0) or 0,
                "volume": vol
            })
        
        # Top 10 by absolute change
        movers = sorted(perp_pairs, key=lambda x: abs(x["change"]), reverse=True)[:10]
        
        lines = ["TOP 10 MOVERS RIGHT NOW\n"]
        for i, m in enumerate(movers, 1):
            emoji = "🟢" if m["change"] > 0 else "🔴"
            lines.append(
                f"{i}. {emoji} {m['symbol']}: ${m['price']:,.4f} "
                f"({m['change']:+.1f}%) "
                f"vol: ${m['volume']:,.0f}"
            )
        
        await message.reply("\n".join(lines))
    
    except Exception as e:
        await message.reply(f"Error: {str(e)}")


@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    """Bot health check"""
    if str(message.from_user.id) != TELEGRAM_ADMIN_ID:
        return
    
    status_lines = ["BOT STATUS\n"]
    
    # Binance check
    try:
        exchange.fetch_ticker("BTC/USDT:USDT")
        status_lines.append("Binance API: connected")
    except:
        status_lines.append("Binance API: DISCONNECTED")
    
    # DeepSeek check
    try:
        test = ask_deepseek("Reply with: OK")
        if test:
            status_lines.append("DeepSeek API: connected")
        else:
            status_lines.append("DeepSeek API: NO RESPONSE")
    except:
        status_lines.append("DeepSeek API: DISCONNECTED")
    
    # Last scan
    if last_scan_time:
        status_lines.append(f"Last scan: {last_scan_time.strftime('%Y-%m-%d %H:%M UTC')}")
    else:
        status_lines.append("Last scan: never")
    
    # Auto-scan
    status_lines.append(f"Auto-scan: {'ON (every 15 min)' if autoscan_enabled else 'OFF'}")
    
    # Currently scanning
    status_lines.append(f"Currently scanning: {'yes' if is_scanning else 'no'}")
    
    await message.reply("\n".join(status_lines))


@dp.message(Command("autoscan"))
async def cmd_autoscan(message: types.Message, command: CommandObject):
    """Toggle auto-scan every 15 minutes"""
    global autoscan_enabled
    
    if str(message.from_user.id) != TELEGRAM_ADMIN_ID:
        return
    
    args = (command.args or "").strip().lower()
    
    if args == "on":
        autoscan_enabled = True
        # Schedule: run at minute 1, 16, 31, 46 (1 min after each 15m candle close)
        if not scheduler.get_job("autoscan"):
            scheduler.add_job(
                auto_scan_job,
                "cron",
                minute="1,16,31,46",
                id="autoscan"
            )
        await message.reply(
            "Auto-scan: ON\n"
            "Scanning every 15 minutes (1 min after candle close)\n"
            "Schedule: xx:01, xx:16, xx:31, xx:46 UTC"
        )
    
    elif args == "off":
        autoscan_enabled = False
        job = scheduler.get_job("autoscan")
        if job:
            job.remove()
        await message.reply("Auto-scan: OFF")
    
    else:
        status = "ON" if autoscan_enabled else "OFF"
        await message.reply(
            f"Auto-scan is currently: {status}\n"
            "Use /autoscan on or /autoscan off"
        )


# ============================================
# AUTO-SCAN JOB (runs on schedule)
# ============================================
async def auto_scan_job():
    """Called by scheduler every 15 minutes"""
    if not autoscan_enabled:
        return
    
    result = await run_full_scan(mode="all")
    
    # Only send to Telegram if there's an actual setup
    # (don't spam with "no setup" every 15 min)
    has_setup = any(word in result.upper() for word in ["LONG", "SHORT"])
    
    if has_setup:
        result = "AUTO-SCAN (15m candle close)\n\n" + result
        await send_long_message(TELEGRAM_ADMIN_ID, result)


# ============================================
# HELPER: Send long messages (split at 4000 chars)
# ============================================
async def send_long_message(chat_id, text):
    """Split and send messages over 4000 chars"""
    if len(text) <= 4000:
        await bot.send_message(chat_id=chat_id, text=text)
        return
    
    # Split at --- separators first
    sections = text.split("---")
    chunks = []
    current = ""
    
    for section in sections:
        if len(current) + len(section) + 5 > 3900:
            if current.strip():
                chunks.append(current.strip())
            current = section
        else:
            current += ("\n---\n" + section) if current else section
    
    if current.strip():
        chunks.append(current.strip())
    
    # If still too long (no --- separators), force split
    final_chunks = []
    for chunk in chunks:
        while len(chunk) > 4000:
            # Split at last newline before 4000
            split_at = chunk.rfind("\n", 0, 4000)
            if split_at == -1:
                split_at = 4000
            final_chunks.append(chunk[:split_at])
            chunk = chunk[split_at:]
        if chunk.strip():
            final_chunks.append(chunk)
    
    for chunk in final_chunks:
        await bot.send_message(chat_id=chat_id, text=chunk)
        await asyncio.sleep(0.3)  # Small delay between messages


# ============================================
# RUN THE BOT
# ============================================
async def main():
    # Start scheduler (for auto-scan)
    scheduler.start()
    
    print("Bot starting...")
    print(f"Admin ID: {TELEGRAM_ADMIN_ID}")
    print("Waiting for Telegram commands...")
    
    # Start polling for Telegram messages
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## HOW IT WORKS

```
Bot starts → sits idle, waiting for Telegram messages
                    |
    User sends /scan  →  bot replies "Scanning..."
                    |        → fetches pairs from Binance
                    |        → runs TA on each
                    |        → sends to DeepSeek
                    |        → replies with setups
                    |
    User sends /scan BTC  →  deep scan single pair
                    |
    User sends /scan meme  →  only volatile movers
                    |
    User sends /top  →  quick top 10 movers (no DeepSeek, instant)
                    |
    User sends /autoscan on  →  bot also scans every 15 min automatically
                                 and sends results if there's a setup
```

The bot is ALWAYS RUNNING. You never need to touch the terminal. Everything happens through Telegram.

---

## IMPORTANT: THE BOT MUST STAY ALIVE

The bot process needs to run 24/7. It's not a script you run once — it's a server.

### Option 1: Run with screen/tmux on VPS

```bash
# SSH into your VPS
tmux new -s scalpbot
python main.py
# Press Ctrl+B then D to detach
# Bot keeps running even after you disconnect
```

### Option 2: Run with systemd (recommended for production)

Create file `/etc/systemd/system/scalpbot.service`:

```ini
[Unit]
Description=Crypto Scalp Scanner Bot
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/your/bot
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl enable scalpbot
sudo systemctl start scalpbot

# Check if running
sudo systemctl status scalpbot

# View logs
journalctl -u scalpbot -f
```

### Option 3: Docker (if already containerized)

```bash
docker compose up -d
# Bot runs in background, restarts automatically
```

---

## ASYNC NOTE FOR THE PROGRAMMER

The scanner functions (fetch from Binance, calculate TA) are synchronous (blocking). The Telegram bot (aiogram) is async. You need to run the scanner in a thread pool so it doesn't block the bot.

Wrap the scan call like this:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

async def run_full_scan(mode="all", specific_pair=None):
    """Run the blocking scanner in a thread pool"""
    loop = asyncio.get_event_loop()
    
    # Run blocking scan in thread
    result = await loop.run_in_executor(
        executor,
        _blocking_scan,  # The actual scan function (synchronous)
        mode,
        specific_pair
    )
    return result

def _blocking_scan(mode, specific_pair):
    """This is the synchronous version that does the actual work"""
    # ... all the Binance fetching, TA calculation, DeepSeek call ...
    # ... same code as before but NOT async ...
    return result_text
```

This way the bot still responds to other commands while a scan is running.

---

## QUICK REFERENCE: USER FLOW

```
You open Telegram
  → /scan
  → Bot: "Scanning top 50 pairs... ~60 seconds"
  → (bot fetches data, calculates TA, sends to DeepSeek)
  → Bot replies with 1-3 trade setups

  → /scan PEPE
  → Bot: "Deep scanning PEPE... ~15 seconds"
  → Bot replies with detailed PEPE/USDT:USDT analysis

  → /scan meme
  → Bot: "Scanning volatile/meme pairs... ~30 seconds"
  → Bot replies with setups from today's biggest movers

  → /top
  → Bot instantly replies with top 10 movers (no DeepSeek wait)

  → /autoscan on
  → Bot scans every 15 min and ONLY messages you if there's a setup
```

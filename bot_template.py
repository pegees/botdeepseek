import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import aiohttp
import json
from datetime import datetime

# ============== CONFIGURATION ==============
# ⚠️ REPLACE THESE WITH YOUR OWN KEYS
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Binance Futures top pairs to scan
PAIRS_TO_SCAN = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "MATICUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT"
]

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ============== BINANCE DATA FETCHING ==============
async def fetch_klines(session, symbol: str, interval: str = "15m", limit: int = 50):
    """Fetch candlestick data from Binance Futures"""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return {"symbol": symbol, "klines": data}
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
    return None


async def fetch_ticker(session, symbol: str):
    """Fetch 24h ticker data"""
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    params = {"symbol": symbol}
    
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
    except Exception as e:
        logger.error(f"Error fetching ticker {symbol}: {e}")
    return None


async def fetch_funding_rate(session, symbol: str):
    """Fetch current funding rate"""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": 1}
    
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data:
                    return data[0]
    except Exception as e:
        logger.error(f"Error fetching funding {symbol}: {e}")
    return None


async def get_market_data(interval: str = "15m"):
    """Fetch all market data for analysis"""
    async with aiohttp.ClientSession() as session:
        # Fetch klines for all pairs
        kline_tasks = [fetch_klines(session, pair, interval) for pair in PAIRS_TO_SCAN]
        ticker_tasks = [fetch_ticker(session, pair) for pair in PAIRS_TO_SCAN]
        funding_tasks = [fetch_funding_rate(session, pair) for pair in PAIRS_TO_SCAN]
        
        klines_results = await asyncio.gather(*kline_tasks)
        ticker_results = await asyncio.gather(*ticker_tasks)
        funding_results = await asyncio.gather(*funding_tasks)
        
        market_data = []
        for i, pair in enumerate(PAIRS_TO_SCAN):
            kline_data = klines_results[i]
            ticker_data = ticker_results[i]
            funding_data = funding_results[i]
            
            if kline_data and ticker_data:
                # Process klines into readable format
                klines = kline_data["klines"]
                recent_candles = []
                for k in klines[-10:]:  # Last 10 candles
                    recent_candles.append({
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5])
                    })
                
                # Calculate some basic metrics
                closes = [c["close"] for c in recent_candles]
                highs = [c["high"] for c in recent_candles]
                lows = [c["low"] for c in recent_candles]
                volumes = [c["volume"] for c in recent_candles]
                
                current_price = closes[-1]
                recent_high = max(highs)
                recent_low = min(lows)
                avg_volume = sum(volumes) / len(volumes)
                last_volume = volumes[-1]
                volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1
                
                # Price change
                price_change_pct = ((closes[-1] - closes[0]) / closes[0]) * 100
                
                market_data.append({
                    "symbol": pair,
                    "price": current_price,
                    "price_change_pct": round(price_change_pct, 2),
                    "volume_24h": float(ticker_data.get("quoteVolume", 0)),
                    "volume_ratio": round(volume_ratio, 2),
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "funding_rate": float(funding_data["fundingRate"]) * 100 if funding_data else 0,
                    "recent_candles": recent_candles[-5:]  # Last 5 for context
                })
        
        return market_data


# ============== DEEPSEEK ANALYSIS ==============
async def analyze_with_deepseek(market_data: list, timeframe: str, style: str = "scalp"):
    """Send market data to DeepSeek for analysis"""
    
    # Build the prompt
    prompt = f"""You are a professional crypto scalp trader. Analyze the following market data and find the BEST scalp setup.

TIMEFRAME: {timeframe}
STYLE: {style} (quick entries, tight stops)

MARKET DATA:
{json.dumps(market_data, indent=2)}

ANALYSIS CRITERIA:
1. Look for liquidity sweeps (price took out recent high/low then reversed)
2. Volume confirmation (volume_ratio > 1.2 suggests interest)
3. Clean support/resistance levels
4. Funding rate extremes can signal reversals
5. Avoid choppy/ranging pairs with no clear direction

RESPOND IN THIS EXACT FORMAT:
PAIR: [symbol]
DIRECTION: [LONG or SHORT]
ENTRY: [price]
STOP LOSS: [price]
TAKE PROFIT: [price]
RISK/REWARD: [ratio like 1:2]
CONFIDENCE: [LOW/MEDIUM/HIGH]
REASONING: [2-3 sentences max explaining the setup]

If no good setup exists, say "NO CLEAR SETUP" and explain why.

Pick the single best opportunity right now."""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a professional crypto trader specializing in scalping. Be concise and precise."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logger.error(f"DeepSeek API error: {response.status} - {error_text}")
                    return f"Error from DeepSeek API: {response.status}"
        except Exception as e:
            logger.error(f"DeepSeek request failed: {e}")
            return f"Failed to connect to DeepSeek: {str(e)}"


# ============== TELEGRAM HANDLERS ==============
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_message = """🤖 *Scalp Signal Bot*

I analyze Binance Futures markets and find scalp setups using AI.

*Commands:*
/scalp - Get a scalp signal (default 15m)
/scalp 5m - Get a signal on 5m timeframe
/scalp 1h - Get a signal on 1h timeframe

*Or just message me:*
"give me a setup"
"find me a long"
"any good shorts?"

I'll scan top 20 futures pairs and find the best opportunity.

⚠️ *Disclaimer:* This is not financial advice. Always manage your risk."""

    await update.message.reply_text(welcome_message, parse_mode="Markdown")


async def get_scalp_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /scalp command"""
    # Parse timeframe from args
    timeframe = "15m"
    if context.args:
        tf = context.args[0].lower()
        if tf in ["1m", "3m", "5m", "15m", "30m", "1h", "4h"]:
            timeframe = tf
    
    await update.message.reply_text(f"🔍 Scanning markets on {timeframe}...\nThis takes ~10 seconds.")
    
    try:
        # Fetch market data
        market_data = await get_market_data(timeframe)
        
        if not market_data:
            await update.message.reply_text("❌ Failed to fetch market data. Try again.")
            return
        
        # Analyze with DeepSeek
        analysis = await analyze_with_deepseek(market_data, timeframe)
        
        # Format and send response
        response = f"""📊 *Scalp Signal ({timeframe})*
        
{analysis}

⏰ Generated: {datetime.now().strftime("%H:%M:%S UTC")}
⚠️ DYOR - Not financial advice"""

        await update.message.reply_text(response, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Error in get_scalp_signal: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle natural language messages"""
    text = update.message.text.lower()
    
    # Detect intent
    scalp_keywords = ["scalp", "setup", "signal", "trade", "entry", "long", "short", "pair", "find"]
    
    if any(kw in text for kw in scalp_keywords):
        # Detect timeframe from message
        timeframe = "15m"  # default
        if "1m" in text:
            timeframe = "1m"
        elif "5m" in text:
            timeframe = "5m"
        elif "30m" in text:
            timeframe = "30m"
        elif "1h" in text or "1 hour" in text:
            timeframe = "1h"
        elif "4h" in text or "4 hour" in text:
            timeframe = "4h"
        
        context.args = [timeframe]
        await get_scalp_signal(update, context)
    else:
        await update.message.reply_text(
            "Not sure what you need. Try:\n"
            "• /scalp - Get a scalp signal\n"
            "• 'give me a setup'\n"
            "• 'find me a long on 5m'"
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")


# ============== MAIN ==============
def main():
    """Start the bot"""
    print("🚀 Starting Scalp Signal Bot...")
    
    # Create application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scalp", get_scalp_signal))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    # Start polling
    print("✅ Bot is running! Send /start to your bot on Telegram.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

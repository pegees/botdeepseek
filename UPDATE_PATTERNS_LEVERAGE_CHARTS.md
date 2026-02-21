# UPDATE: Chart Patterns + Leverage Gains + TradingView Screenshots

> **Goal:** Make the bot output look like the "AI FUTURE SIGNAL" bot — chart pattern detection, leveraged % gain display, and TradingView chart screenshots attached to every signal.

---

## FEATURE 1: CHART PATTERN DETECTION

### What It Does

Before sending data to DeepSeek, detect common chart patterns from the candle data. This gives DeepSeek better context AND gives a clean label for the signal message (e.g. "Chart Pattern: Ranging Channel").

### Patterns to Detect (15m candles)

```python
"""
CHART PATTERN DETECTOR
======================
Input: DataFrame with OHLCV + indicators (from analyze_pair)
Output: { "pattern": str, "confidence": int, "direction_bias": str }

Install scipy if not already: pip install scipy
"""

import numpy as np
from scipy.signal import argrelextrema


def detect_chart_pattern(df, lookback=50):
    """
    Analyze last N candles to detect chart patterns.
    Returns the strongest pattern found.
    """
    recent = df.tail(lookback).copy()
    close = recent["close"].values
    high = recent["high"].values
    low = recent["low"].values
    
    # Find swing highs and swing lows (local extrema)
    # order=5 means compare with 5 neighbors on each side
    swing_high_idx = argrelextrema(high, np.greater, order=5)[0]
    swing_low_idx = argrelextrema(low, np.less, order=5)[0]
    
    swing_highs = [(i, high[i]) for i in swing_high_idx]
    swing_lows = [(i, low[i]) for i in swing_low_idx]
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"pattern": "No Clear Pattern", "confidence": 0, "direction_bias": "neutral"}
    
    # Get the slopes of highs and lows
    # Positive slope = rising, Negative = falling, ~0 = flat
    high_slope = calculate_slope([h[1] for h in swing_highs])
    low_slope = calculate_slope([l[1] for l in swing_lows])
    
    # Price range for threshold calculations
    price_range = max(high) - min(low)
    flat_threshold = price_range * 0.02  # 2% of range = "flat"
    
    pattern = identify_pattern(high_slope, low_slope, swing_highs, swing_lows, 
                                close, high, low, flat_threshold, price_range)
    
    return pattern


def calculate_slope(values):
    """Linear regression slope of a series of values"""
    if len(values) < 2:
        return 0
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    return slope


def identify_pattern(high_slope, low_slope, swing_highs, swing_lows, 
                      close, high, low, flat_threshold, price_range):
    """
    Identify pattern based on swing high/low slopes and structure.
    """
    
    # Normalize slopes relative to price range
    hs = high_slope / price_range if price_range > 0 else 0
    ls = low_slope / price_range if price_range > 0 else 0
    
    current_price = close[-1]
    last_high = swing_highs[-1][1]
    last_low = swing_lows[-1][1]
    
    # =========================================
    # RANGING CHANNEL (horizontal)
    # Highs flat, lows flat — price bouncing between support and resistance
    # =========================================
    if abs(hs) < 0.05 and abs(ls) < 0.05:
        # Check if price is near top or bottom of range
        range_pos = (current_price - min(low)) / price_range
        if range_pos > 0.8:
            bias = "short"  # At top of range, expect rejection
        elif range_pos < 0.2:
            bias = "long"   # At bottom of range, expect bounce
        else:
            bias = "neutral"
        
        return {
            "pattern": "Ranging Channel",
            "confidence": 75,
            "direction_bias": bias,
            "description": f"Price ranging between {min(low):.4f} and {max(high):.4f}"
        }
    
    # =========================================
    # ASCENDING CHANNEL
    # Both highs and lows making higher highs/lows (parallel uptrend)
    # =========================================
    if hs > 0.03 and ls > 0.03 and abs(hs - ls) < 0.05:
        return {
            "pattern": "Ascending Channel",
            "confidence": 70,
            "direction_bias": "long",
            "description": "Parallel uptrend channel — buy at lower trendline"
        }
    
    # =========================================
    # DESCENDING CHANNEL
    # Both highs and lows making lower highs/lows
    # =========================================
    if hs < -0.03 and ls < -0.03 and abs(hs - ls) < 0.05:
        return {
            "pattern": "Descending Channel",
            "confidence": 70,
            "direction_bias": "short",
            "description": "Parallel downtrend channel — sell at upper trendline"
        }
    
    # =========================================
    # ASCENDING TRIANGLE
    # Highs flat (resistance), lows rising (higher lows)
    # Bullish — usually breaks up
    # =========================================
    if abs(hs) < 0.03 and ls > 0.05:
        return {
            "pattern": "Ascending Triangle",
            "confidence": 80,
            "direction_bias": "long",
            "description": "Flat resistance + rising support — bullish breakout likely"
        }
    
    # =========================================
    # DESCENDING TRIANGLE
    # Lows flat (support), highs falling (lower highs)
    # Bearish — usually breaks down
    # =========================================
    if abs(ls) < 0.03 and hs < -0.05:
        return {
            "pattern": "Descending Triangle",
            "confidence": 80,
            "direction_bias": "short",
            "description": "Flat support + falling resistance — bearish breakdown likely"
        }
    
    # =========================================
    # SYMMETRICAL TRIANGLE (Converging)
    # Highs falling, lows rising — squeeze
    # =========================================
    if hs < -0.03 and ls > 0.03:
        return {
            "pattern": "Symmetrical Triangle",
            "confidence": 65,
            "direction_bias": "neutral",  # Can break either way
            "description": "Converging trendlines — breakout imminent, direction unclear"
        }
    
    # =========================================
    # RISING WEDGE
    # Both rising but highs rising slower than lows (converging upward)
    # Bearish reversal pattern
    # =========================================
    if hs > 0.02 and ls > 0.02 and ls > hs:
        return {
            "pattern": "Rising Wedge",
            "confidence": 70,
            "direction_bias": "short",
            "description": "Converging uptrend — bearish reversal likely"
        }
    
    # =========================================
    # FALLING WEDGE
    # Both falling but lows falling slower than highs (converging downward)
    # Bullish reversal pattern
    # =========================================
    if hs < -0.02 and ls < -0.02 and ls > hs:
        return {
            "pattern": "Falling Wedge",
            "confidence": 70,
            "direction_bias": "long",
            "description": "Converging downtrend — bullish reversal likely"
        }
    
    # =========================================
    # BROADENING / MEGAPHONE
    # Highs rising, lows falling — expanding range
    # =========================================
    if hs > 0.03 and ls < -0.03:
        return {
            "pattern": "Broadening Formation",
            "confidence": 55,
            "direction_bias": "neutral",
            "description": "Expanding volatility — choppy, trade edges only"
        }
    
    # =========================================
    # DOUBLE TOP (M shape)
    # Two similar highs with a dip in between
    # =========================================
    if len(swing_highs) >= 2:
        h1 = swing_highs[-2][1]
        h2 = swing_highs[-1][1]
        if abs(h1 - h2) / h1 < 0.005:  # Highs within 0.5% of each other
            if current_price < h2 * 0.995:  # Price pulled back from second top
                return {
                    "pattern": "Double Top",
                    "confidence": 75,
                    "direction_bias": "short",
                    "description": f"Two rejections at ~{h1:.4f} — bearish reversal"
                }
    
    # =========================================
    # DOUBLE BOTTOM (W shape)
    # Two similar lows with a bounce in between
    # =========================================
    if len(swing_lows) >= 2:
        l1 = swing_lows[-2][1]
        l2 = swing_lows[-1][1]
        if abs(l1 - l2) / l1 < 0.005:  # Lows within 0.5%
            if current_price > l2 * 1.005:  # Price bounced from second bottom
                return {
                    "pattern": "Double Bottom",
                    "confidence": 75,
                    "direction_bias": "long",
                    "description": f"Two bounces at ~{l1:.4f} — bullish reversal"
                }
    
    # =========================================
    # BULL FLAG
    # Sharp up move followed by slight downward consolidation
    # =========================================
    pre_flag = close[:len(close)//2]
    flag_part = close[len(close)//2:]
    pre_move = (pre_flag[-1] - pre_flag[0]) / pre_flag[0]
    flag_move = (flag_part[-1] - flag_part[0]) / flag_part[0]
    
    if pre_move > 0.02 and -0.015 < flag_move < 0:  # Up move then slight down drift
        return {
            "pattern": "Bull Flag",
            "confidence": 70,
            "direction_bias": "long",
            "description": "Strong rally + pullback consolidation — continuation likely"
        }
    
    # =========================================
    # BEAR FLAG
    # Sharp down move followed by slight upward consolidation
    # =========================================
    if pre_move < -0.02 and 0 < flag_move < 0.015:
        return {
            "pattern": "Bear Flag",
            "confidence": 70,
            "direction_bias": "short",
            "description": "Strong drop + bounce consolidation — continuation down likely"
        }
    
    # No clear pattern
    return {
        "pattern": "No Clear Pattern",
        "confidence": 0,
        "direction_bias": "neutral",
        "description": "No recognizable pattern in last 50 candles"
    }
```

### Integrate Into analyze_pair

Add the pattern detection result to the data that gets sent to DeepSeek:

```python
def analyze_pair(exchange, symbol):
    # ... existing candle fetch and TA code ...
    
    # NEW: Detect chart pattern
    pattern = detect_chart_pattern(df, lookback=50)
    
    # Add to result dict
    result["chart_pattern"] = pattern["pattern"]
    result["pattern_confidence"] = pattern["confidence"]
    result["pattern_bias"] = pattern["direction_bias"]
    result["pattern_description"] = pattern["description"]
    
    return result
```

### Add to DeepSeek Data Dump

In `build_deepseek_prompt`, add this line to each pair summary:

```python
pair_summaries += f"Chart Pattern: {d['chart_pattern']} (confidence: {d['pattern_confidence']}%, bias: {d['pattern_bias']})\n"
pair_summaries += f"Pattern Detail: {d['pattern_description']}\n"
```

---

## FEATURE 2: LEVERAGED GAIN CALCULATION

### What It Does

Show the gains at each TP level based on a configurable leverage. The "AI FUTURE SIGNAL" bot uses 125x. We show the raw % move AND the leveraged % gain.

### Implementation

```python
DEFAULT_LEVERAGE = 50  # Set your default leverage. 125x is extremely risky.

def calculate_leveraged_gains(entry, sl, tp1, tp2, tp3, direction, leverage=None):
    """
    Calculate leveraged % gains and liquidation price.
    
    The "AI FUTURE SIGNAL" bot shows 125x gains to make numbers look big.
    15.2% gain at TP1 is actually only 0.12% raw price move.
    
    We show BOTH so the user knows reality.
    """
    if leverage is None:
        leverage = DEFAULT_LEVERAGE
    
    if direction == "long":
        raw_sl = (sl - entry) / entry * 100
        raw_tp1 = (tp1 - entry) / entry * 100
        raw_tp2 = (tp2 - entry) / entry * 100
        raw_tp3 = (tp3 - entry) / entry * 100
        
        # Liquidation price (approximate — exchanges add maintenance margin)
        # At 125x, you get liquidated if price drops ~0.8% (1/125)
        liquidation = entry * (1 - (1 / leverage) * 0.95)  # 95% of margin = liq
    else:
        raw_sl = (entry - sl) / entry * 100
        raw_tp1 = (entry - tp1) / entry * 100
        raw_tp2 = (entry - tp2) / entry * 100
        raw_tp3 = (entry - tp3) / entry * 100
        
        liquidation = entry * (1 + (1 / leverage) * 0.95)
    
    # Negative because SL is a loss
    raw_sl = -abs(raw_sl)
    
    return {
        "leverage": leverage,
        "sl_raw": round(raw_sl, 2),
        "sl_leveraged": round(raw_sl * leverage, 1),
        "tp1_raw": round(raw_tp1, 2),
        "tp1_leveraged": round(raw_tp1 * leverage, 1),
        "tp2_raw": round(raw_tp2, 2),
        "tp2_leveraged": round(raw_tp2 * leverage, 1),
        "tp3_raw": round(raw_tp3, 2),
        "tp3_leveraged": round(raw_tp3 * leverage, 1),
        "liquidation_price": round(liquidation, 8)
    }
```

### Update Signal Output Format

Tell DeepSeek to include leverage calculations, or calculate them yourself after DeepSeek returns the setup and append to the message.

Better approach: calculate AFTER DeepSeek returns the trade plan, then append:

```python
def format_signal_with_leverage(deepseek_result, leverage=50):
    """
    Parse DeepSeek's response, extract entry/SL/TP numbers,
    calculate leveraged gains, and append to the message.
    """
    import re
    
    # Try to extract prices from DeepSeek response
    entry_match = re.search(r'Entry[:\s]*\$?([\d,.]+)', deepseek_result)
    sl_match = re.search(r'Stop Loss[:\s]*\$?([\d,.]+)', deepseek_result)
    tp1_match = re.search(r'TP1[:\s]*\$?([\d,.]+)', deepseek_result)
    tp2_match = re.search(r'TP2[:\s]*\$?([\d,.]+)', deepseek_result)
    tp3_match = re.search(r'TP3[:\s]*\$?([\d,.]+)', deepseek_result)
    
    if not all([entry_match, sl_match, tp1_match]):
        # Can't parse — return original
        return deepseek_result
    
    entry = float(entry_match.group(1).replace(",", ""))
    sl = float(sl_match.group(1).replace(",", ""))
    tp1 = float(tp1_match.group(1).replace(",", ""))
    tp2 = float(tp2_match.group(1).replace(",", "")) if tp2_match else None
    tp3 = float(tp3_match.group(1).replace(",", "")) if tp3_match else None
    
    # Detect direction
    direction = "long" if tp1 > entry else "short"
    
    gains = calculate_leveraged_gains(entry, sl, tp1, tp2 or tp1, tp3 or tp1, direction, leverage)
    
    # Append leverage info
    leverage_section = f"""
{'─' * 30}
📊 LEVERAGE CALCULATION ({leverage}x)

At {leverage}x leverage:
- SL hit: {gains['sl_leveraged']}% loss (raw: {gains['sl_raw']}%)
- TP1 hit: +{gains['tp1_leveraged']}% gain (raw: +{gains['tp1_raw']}%)"""
    
    if tp2:
        leverage_section += f"\n- TP2 hit: +{gains['tp2_leveraged']}% gain (raw: +{gains['tp2_raw']}%)"
    if tp3:
        leverage_section += f"\n- TP3 hit: +{gains['tp3_leveraged']}% gain (raw: +{gains['tp3_raw']}%)"
    
    leverage_section += f"""
- Liquidation: ${gains['liquidation_price']:,.4f}

{chr(9888)} {leverage}x leverage = liquidation at ~{abs(round(1/leverage*100*0.95, 2))}% against you
"""
    
    return deepseek_result + leverage_section
```

### Usage in the Main Pipeline

```python
# After getting DeepSeek result:
analysis = ask_deepseek(prompt)

# Add leverage calculations
analysis_with_leverage = format_signal_with_leverage(analysis, leverage=50)

# Send to Telegram
send_to_telegram(bot_token, channel_id, analysis_with_leverage)
```

### Add /leverage Command to Telegram Bot

```python
# Let user change leverage from Telegram
user_leverage = 50  # Default

@dp.message(Command("leverage"))
async def cmd_leverage(message: types.Message, command: CommandObject):
    global user_leverage
    
    args = (command.args or "").strip()
    
    if not args:
        await message.reply(f"Current leverage: {user_leverage}x\nUse /leverage 50 to change")
        return
    
    try:
        new_lev = int(args)
        if new_lev < 1 or new_lev > 125:
            await message.reply("Leverage must be between 1 and 125")
            return
        user_leverage = new_lev
        
        liq_pct = round(1 / new_lev * 100 * 0.95, 2)
        await message.reply(
            f"Leverage set to {new_lev}x\n"
            f"Liquidation distance: ~{liq_pct}% against you\n"
            f"{'WARNING: Extremely high leverage!' if new_lev > 50 else ''}"
        )
    except ValueError:
        await message.reply("Invalid number. Use /leverage 50")
```

---

## FEATURE 3: TRADINGVIEW CHART SCREENSHOTS

### What It Does

Attach a TradingView chart image to every signal. The "AI FUTURE SIGNAL" bot shows a chart with ZigZag, Bollinger Bands, RSI, and MACD visible. We generate chart screenshots using TradingView's free widget/image API.

### Method 1: TradingView Mini Chart Image URL (Simplest)

TradingView has a public chart image endpoint. No API key needed.

```python
def get_tradingview_chart_url(symbol, interval="15", width=800, height=500):
    """
    Generate a TradingView chart image URL.
    
    This uses TradingView's mini chart widget screenshot service.
    
    symbol format:
    - Crypto on Binance: "BINANCE:BTCUSDT.P" (.P for perpetual)
    - Gold: "OANDA:XAUUSD"
    - Stock: "NASDAQ:AAPL"
    
    interval: "1", "5", "15", "60", "240", "D"
    """
    # Convert ccxt symbol to TradingView format
    # "BTC/USDT:USDT" -> "BINANCE:BTCUSDT.P"
    base = symbol.split("/")[0]
    tv_symbol = f"BINANCE:{base}USDT.P"
    
    # TradingView chart image URL
    # This generates a static chart image
    url = (
        f"https://s3.tradingview.com/widgetembed/"
        f"?frameElementId=tradingview_chart"
        f"&symbol={tv_symbol}"
        f"&interval={interval}"
        f"&hidesidetoolbar=1"
        f"&symboledit=0"
        f"&saveimage=0"
        f"&toolbarbg=f1f3f6"
        f"&studies=RSI@tv-basicstudies"
        f"&studies=MACD@tv-basicstudies"
        f"&studies=BB@tv-basicstudies"
        f"&theme=dark"
        f"&style=1"
        f"&timezone=Etc/UTC"
        f"&withdateranges=0"
        f"&width={width}"
        f"&height={height}"
        f"&utm_source=scalp_bot"
    )
    return url
```

### Method 2: Screenshot with Selenium/Playwright (Best Quality)

For actual chart screenshots like the "AI FUTURE SIGNAL" bot uses, you need to render the TradingView widget and take a screenshot.

```python
"""
TRADINGVIEW CHART SCREENSHOT GENERATOR
Install: pip install playwright
Setup:   playwright install chromium
"""

import asyncio
from playwright.async_api import async_playwright
import os


async def capture_tradingview_chart(symbol, interval="15", save_path="chart.png"):
    """
    Open TradingView chart in headless browser, add indicators, take screenshot.
    
    symbol: ccxt format like "BTC/USDT:USDT"
    interval: "1", "5", "15", "60", "240"
    save_path: where to save the PNG
    """
    
    # Convert symbol to TradingView format
    base = symbol.split("/")[0]
    tv_symbol = f"BINANCE:{base}USDT.P"
    
    # TradingView advanced chart widget HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; background: #1a1a2e; }}
            #chart {{ width: 900px; height: 600px; }}
        </style>
    </head>
    <body>
        <div id="chart"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({{
            "autosize": false,
            "width": 900,
            "height": 600,
            "symbol": "{tv_symbol}",
            "interval": "{interval}",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#1a1a2e",
            "enable_publishing": false,
            "hide_side_toolbar": true,
            "allow_symbol_change": false,
            "save_image": false,
            "container_id": "chart",
            "studies": [
                "RSI@tv-basicstudies",
                "MACD@tv-basicstudies",
                "BB@tv-basicstudies"
            ]
        }});
        </script>
    </body>
    </html>
    """
    
    # Save temp HTML
    temp_html = "/tmp/tv_chart.html"
    with open(temp_html, "w") as f:
        f.write(html_content)
    
    # Render with Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 920, "height": 620})
        
        await page.goto(f"file://{temp_html}")
        
        # Wait for chart to fully load
        await page.wait_for_timeout(8000)  # TradingView widget needs time
        
        # Take screenshot
        await page.screenshot(path=save_path, full_page=False)
        await browser.close()
    
    # Cleanup
    os.remove(temp_html)
    
    return save_path


# Synchronous wrapper for non-async code
def capture_chart_sync(symbol, interval="15", save_path="chart.png"):
    return asyncio.run(capture_tradingview_chart(symbol, interval, save_path))
```

### Method 3: Generate Charts with mplfinance (No Browser Needed)

If you don't want to deal with Playwright/Selenium, generate charts directly from the candle data using mplfinance. Faster and no browser dependency.

```python
"""
CHART GENERATOR using mplfinance
Install: pip install mplfinance
"""

import mplfinance as mpf
import pandas as pd


def generate_chart_image(df, symbol, pattern_name="", save_path="chart.png"):
    """
    Generate a candlestick chart with indicators from raw OHLCV data.
    
    df: DataFrame with columns [timestamp, open, high, low, close, volume]
         + calculated indicators (ema9, ema21, rsi, etc.)
    symbol: pair name for title
    save_path: where to save PNG
    """
    
    # Prepare DataFrame for mplfinance (needs DatetimeIndex)
    chart_df = df.tail(80).copy()  # Last 80 candles
    chart_df.set_index("timestamp", inplace=True)
    chart_df.index.name = "Date"
    
    # Additional plots (EMAs, Bollinger Bands)
    add_plots = []
    
    # EMA 9 (yellow line)
    if "ema9" in chart_df.columns:
        add_plots.append(mpf.make_addplot(chart_df["ema9"], color="yellow", width=1))
    
    # EMA 21 (orange line)
    if "ema21" in chart_df.columns:
        add_plots.append(mpf.make_addplot(chart_df["ema21"], color="orange", width=1))
    
    # EMA 50 (blue line)
    if "ema50" in chart_df.columns:
        add_plots.append(mpf.make_addplot(chart_df["ema50"], color="cyan", width=1))
    
    # Bollinger Bands
    if "BBU_20_2.0" in chart_df.columns and "BBL_20_2.0" in chart_df.columns:
        add_plots.append(mpf.make_addplot(chart_df["BBU_20_2.0"], color="gray", width=0.7, linestyle="--"))
        add_plots.append(mpf.make_addplot(chart_df["BBL_20_2.0"], color="gray", width=0.7, linestyle="--"))
    
    # RSI in separate panel
    if "rsi" in chart_df.columns:
        add_plots.append(mpf.make_addplot(chart_df["rsi"], panel=2, color="magenta", 
                                           ylabel="RSI", ylim=(0, 100)))
        # RSI 30/70 lines
        rsi_30 = pd.Series(30, index=chart_df.index)
        rsi_70 = pd.Series(70, index=chart_df.index)
        add_plots.append(mpf.make_addplot(rsi_30, panel=2, color="green", width=0.5, linestyle="--"))
        add_plots.append(mpf.make_addplot(rsi_70, panel=2, color="red", width=0.5, linestyle="--"))
    
    # MACD in separate panel
    if "MACDh_12_26_9" in chart_df.columns:
        macd_hist = chart_df["MACDh_12_26_9"]
        colors = ["green" if v >= 0 else "red" for v in macd_hist]
        add_plots.append(mpf.make_addplot(macd_hist, panel=3, type="bar", color=colors, ylabel="MACD"))
        
        if "MACD_12_26_9" in chart_df.columns:
            add_plots.append(mpf.make_addplot(chart_df["MACD_12_26_9"], panel=3, color="blue", width=0.8))
        if "MACDs_12_26_9" in chart_df.columns:
            add_plots.append(mpf.make_addplot(chart_df["MACDs_12_26_9"], panel=3, color="orange", width=0.8))
    
    # Chart style (dark theme like TradingView)
    mc = mpf.make_marketcolors(
        up="green", down="red",
        edge="inherit",
        wick="inherit",
        volume="in",
        ohlc="inherit"
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        base_mpf_style="nightclouds",
        rc={"font.size": 8}
    )
    
    # Title
    title = f"{symbol} — 15m"
    if pattern_name:
        title += f" | {pattern_name}"
    
    # Render chart
    fig, axes = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        volume=True,
        addplot=add_plots if add_plots else None,
        title=title,
        figsize=(12, 8),
        panel_ratios=(4, 1, 1, 1) if "MACDh_12_26_9" in chart_df.columns else (4, 1, 1),
        returnfig=True,
        savefig=save_path
    )
    
    return save_path
```

### Send Chart Image to Telegram

```python
async def send_signal_with_chart(chat_id, signal_text, chart_path):
    """Send the chart image first, then the signal text as caption or follow-up"""
    
    from aiogram.types import FSInputFile
    
    if chart_path and os.path.exists(chart_path):
        photo = FSInputFile(chart_path)
        
        # If signal text fits in caption (1024 char limit for photos)
        if len(signal_text) <= 1024:
            await bot.send_photo(chat_id=chat_id, photo=photo, caption=signal_text)
        else:
            # Send chart first, then text as separate message
            await bot.send_photo(chat_id=chat_id, photo=photo, caption="Chart — 15m")
            await send_long_message(chat_id, signal_text)
        
        # Cleanup
        os.remove(chart_path)
    else:
        # No chart available, send text only
        await send_long_message(chat_id, signal_text)
```

---

## UPDATED SIGNAL OUTPUT FORMAT

With all 3 features combined, the signal message should look like this:

```
[CHART IMAGE ATTACHED]

🟢 LONG — PEPE/USDT:USDT
Confidence: 8/10

📐 Chart Pattern: Ascending Triangle (80% confidence)
   Flat resistance + rising support — bullish breakout likely

📊 Confluence:
- RSI(14): 32.1 — near oversold, bounce zone
- EMA: 9 crossing above 21 (fresh bullish cross)
- Volume: 2.1x average — real participation
- MACD: histogram just flipped positive
- Bollinger: near lower band (22%) — mean reversion setup
- Funding: -0.02% — shorts crowded, contrarian long edge

💰 Trade Plan:
- Entry: $0.00001245
- Stop Loss: $0.00001220 (-2.0%)
- TP1: $0.00001275 (+2.4%) — close 40%, move SL to BE
- TP2: $0.00001310 (+5.2%) — close 40%
- TP3: $0.00001370 (+10.0%) — close 20% runner
- R:R: 1:2.5

──────────────────────────────
📊 LEVERAGE CALCULATION (50x)

At 50x leverage:
- SL hit: -100.0% loss (raw: -2.0%)
- TP1 hit: +120.0% gain (raw: +2.4%)
- TP2 hit: +260.0% gain (raw: +5.2%)
- TP3 hit: +500.0% gain (raw: +10.0%)
- Liquidation: $0.00001221

⚠ 50x leverage = liquidation at ~1.9% against you

⚠ Invalidation: Break below $0.00001210 with volume kills this
```

---

## UPDATED PIPELINE (Full Flow)

```python
async def run_full_scan(mode="all", specific_pair=None):
    # 1. Fetch pairs
    top_pairs = get_scannable_pairs(exchange, top_n=50)
    
    # 2. Analyze each pair (TA + chart pattern)
    all_data = []
    for pair_info in top_pairs:
        analysis = analyze_pair(exchange, pair_info["symbol"])
        if analysis:
            # Add chart pattern detection
            pattern = detect_chart_pattern(df_cache[pair_info["symbol"]])
            analysis["chart_pattern"] = pattern["pattern"]
            analysis["pattern_confidence"] = pattern["confidence"]
            analysis["pattern_bias"] = pattern["direction_bias"]
            analysis["pattern_description"] = pattern["description"]
            all_data.append(analysis)
    
    # 3. Send to DeepSeek
    prompt = build_deepseek_prompt(all_data)
    deepseek_result = ask_deepseek(prompt)
    
    # 4. Add leverage calculations
    result_with_leverage = format_signal_with_leverage(deepseek_result, leverage=user_leverage)
    
    # 5. Generate chart for the recommended pair
    # Parse which pair DeepSeek picked
    recommended_pair = extract_pair_from_result(deepseek_result)
    chart_path = None
    if recommended_pair:
        # Option A: mplfinance (fast, no browser)
        chart_path = generate_chart_image(
            df_cache[recommended_pair],
            recommended_pair,
            pattern_name=get_pattern_for_pair(recommended_pair, all_data),
            save_path=f"/tmp/chart_{recommended_pair.replace('/', '_')}.png"
        )
        
        # Option B: TradingView screenshot (prettier but slower)
        # chart_path = await capture_tradingview_chart(recommended_pair, save_path=f"/tmp/chart.png")
    
    # 6. Send to Telegram with chart
    await send_signal_with_chart(TELEGRAM_ADMIN_ID, result_with_leverage, chart_path)


def extract_pair_from_result(deepseek_result):
    """Parse DeepSeek response to find which pair was recommended"""
    import re
    # Look for patterns like "LONG — BTC/USDT:USDT" or "SHORT — PEPE/USDT"
    match = re.search(r'(?:LONG|SHORT)\s*(?:—|-)\s*(\w+/USDT(?::USDT)?)', deepseek_result)
    if match:
        return match.group(1)
    return None
```

---

## DEPENDENCIES TO ADD

```bash
pip install scipy           # For chart pattern detection (argrelextrema)
pip install mplfinance      # For chart image generation (Method 3)

# Only if using TradingView screenshots (Method 2):
pip install playwright
playwright install chromium
```

---

## SUMMARY

| Feature | What it adds |
|---|---|
| Chart pattern detection | Scans swing highs/lows to identify 12 patterns (channel, triangle, wedge, flag, double top/bottom, etc.) |
| Leveraged gain calc | Shows TP/SL % at user's chosen leverage (default 50x) + liquidation price |
| Chart screenshots | Generates candlestick chart with BB, RSI, MACD as PNG, attaches to Telegram signal |
| /leverage command | User can change leverage display from Telegram |

The bot now outputs signals that look like the "AI FUTURE SIGNAL" bot but with real data backing every call.

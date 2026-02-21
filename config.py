"""
Centralized configuration for the Scalp Signal Bot.
"""
import os
import sys
import logging
from typing import Set, List
from logging.handlers import TimedRotatingFileHandler

# ============== API CONFIGURATION ==============
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Binance API (required for authenticated requests)
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1"


def validate_config():
    """Validate required configuration. Call this before starting the bot."""
    missing = []
    if not TELEGRAM_TOKEN:
        missing.append("TELEGRAM_TOKEN")
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not BINANCE_API_KEY:
        missing.append("BINANCE_API_KEY")
    if not BINANCE_API_SECRET:
        missing.append("BINANCE_API_SECRET")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please enter your API keys in the launcher and click Save Keys."
        )

# ============== TIMEOUTS (seconds) ==============
BINANCE_TIMEOUT = 15  # Increased for large requests
DEEPSEEK_TIMEOUT = 60  # Increased for analyzing 200+ pairs

# ============== RETRY CONFIGURATION ==============
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # Base delay in seconds (will use exponential backoff)

# ============== BATCHING (for scanning all pairs) ==============
BATCH_SIZE = 50  # Pairs to fetch per batch
BATCH_DELAY = 0.5  # Seconds between batches to avoid rate limits

# ============== RATE LIMITING ==============
USER_RATE_LIMIT = 10  # Max requests per minute per user

# ============== AUTHENTICATION ==============
# Set to False to allow everyone (default for easy setup)
WHITELIST_ENABLED = False

# Add Telegram user IDs here if you enable whitelist
# Get your ID by messaging @userinfobot on Telegram
ALLOWED_USER_IDS: Set[int] = {
    # Example: 123456789, 987654321
}

# ============== TRADING PAIRS ==============
# Fallback pairs if Binance API fails (used only as backup)
FALLBACK_PAIRS: List[str] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "MATICUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT",
    "PEPEUSDT", "SHIBUSDT", "WIFUSDT", "BONKUSDT", "FLOKIUSDT",
]

# ============== TIMEFRAMES ==============
VALID_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h"]
DEFAULT_TIMEFRAME = "15m"

# ============== SCHEDULER ==============
# Hours at which to run scheduled scans (24h format)
ALERT_HOURS = [0, 4, 8, 12, 16, 20]
ALERT_MIN_CONFIDENCE = "MEDIUM"  # Minimum confidence to send alert: LOW, MEDIUM, HIGH

# ============== INDICATOR CONFIGURATION ==============
INDICATOR_PARAMS = {
    "rsi": {"period": 14, "overbought": 70, "oversold": 30},
    "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "ema": {"fast_period": 9, "slow_period": 21},
    "volume": {"ma_period": 20, "spike_threshold": 2.0},
    "cvd": {"lookback": 20},
    "market_structure": {"swing_lookback": 3},
    "support_resistance": {"lookback": 50, "min_touches": 2},
    "liquidity_sweep": {"lookback": 20, "sweep_candles": 3},
    "fvg": {"lookback": 30, "min_gap_pct": 0.1},
    "whale": {"lookback": 50, "whale_threshold": 3.0},
    "open_interest": {"lookback": 20},
}

# ============== SCANNER CONFIGURATION ==============
SCANNER_TIMEFRAMES = ["5m", "15m", "1h"]  # Default timeframes for multi-TF scan
SCANNER_MIN_VOLUME_24H = 5_000_000  # Minimum 24h volume in USD
SCANNER_MIN_CONFLUENCE = 3  # Minimum indicators agreeing for signal
SCANNER_MIN_SCORE = 0.5  # Minimum score (0-1) to include in results
SCANNER_TOP_N = 10  # Number of top results to return

# ============== DISCORD CONFIGURATION ==============
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ============== COOLDOWN CONFIGURATION ==============
COOLDOWN_PER_PAIR_MINUTES = 60  # Minutes between same signal on same pair
COOLDOWN_MAX_ALERTS_PER_HOUR = 10  # Maximum alerts per hour globally

# ============== DASHBOARD CONFIGURATION ==============
DASHBOARD_ENABLED = True
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8080

# ============== PATHS ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATABASE_PATH = os.path.join(DATA_DIR, "signals.db")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ============== LOGGING SETUP ==============
def setup_logging() -> logging.Logger:
    """Configure logging with file and console output."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # File handler (rotating daily, keep 7 days)
    file_handler = TimedRotatingFileHandler(
        os.path.join(LOGS_DIR, "bot.log"),
        when="midnight",
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger

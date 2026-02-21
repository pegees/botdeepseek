"""
Configuration Settings - All Tunable Parameters
================================================
Professional-grade 15m crypto scalping system configuration.
All parameters in one place for easy tuning.
"""
import os
from typing import List, Dict
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# TIMEFRAMES
# =============================================================================
PRIMARY_TIMEFRAME = "15m"
CONFIRMATION_TIMEFRAMES = ["1m", "5m", "1h"]
SCANNER_TIMEFRAME = "15m"


# =============================================================================
# CONFLUENCE LAYER WEIGHTS (Must sum to 1.0)
# =============================================================================
@dataclass
class LayerWeights:
    """4-layer confluence scoring weights."""
    technical: float = 0.35      # TA indicators
    orderflow: float = 0.25      # Order book, trade flow
    onchain: float = 0.20        # Whale, exchange flows
    sentiment: float = 0.10      # Fear/Greed, social
    backtest: float = 0.10       # Historical pattern win rate

    def __post_init__(self):
        total = self.technical + self.orderflow + self.onchain + self.sentiment + self.backtest
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

LAYER_WEIGHTS = LayerWeights()


# =============================================================================
# SIGNAL THRESHOLDS
# =============================================================================
@dataclass
class SignalThresholds:
    """Signal quality thresholds."""
    high_confidence: int = 80     # ⭐⭐⭐⭐ Premium signal
    medium_confidence: int = 65   # ⭐⭐⭐ Standard signal
    low_confidence: int = 50      # ⭐⭐ Weak (usually skip)
    minimum_score: int = 65       # Below this = no signal
    signal_expiry_minutes: int = 15  # Signal valid for 1 candle

SIGNAL_THRESHOLDS = SignalThresholds()


# =============================================================================
# RISK MANAGEMENT
# =============================================================================
@dataclass
class RiskConfig:
    """Position sizing and risk parameters."""
    # Core risk
    risk_per_trade_pct: float = 2.0       # % of account per trade
    max_portfolio_heat: float = 6.0       # Max total risk exposure
    min_rr_ratio: float = 2.0             # Minimum reward:risk ratio

    # ATR-based stops
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5        # SL = entry ± (ATR × 1.5)

    # Multi-level take profits (must sum to 1.0)
    tp1_rr: float = 1.5                   # TP1 at 1.5R
    tp1_size: float = 0.40                # Close 40% at TP1
    tp2_rr: float = 2.5                   # TP2 at 2.5R
    tp2_size: float = 0.40                # Close 40% at TP2
    tp3_rr: float = 4.0                   # TP3 at 4R (runner)
    tp3_size: float = 0.20                # Close 20% at TP3

    # Position limits
    max_concurrent_positions: int = 5
    max_correlation: float = 0.85         # Skip if >85% correlated

    def __post_init__(self):
        tp_total = self.tp1_size + self.tp2_size + self.tp3_size
        if abs(tp_total - 1.0) > 0.001:
            raise ValueError(f"TP sizes must sum to 1.0, got {tp_total}")

RISK_CONFIG = RiskConfig()


# =============================================================================
# CIRCUIT BREAKERS (Anti-Emotion Safeguards)
# =============================================================================
@dataclass
class CircuitBreakers:
    """9 anti-emotion safeguards."""
    # 1. Revenge trade prevention
    revenge_cooldown_seconds: int = 900   # 15min cooldown after SL hit

    # 2. Overtrading prevention
    max_signals_per_hour: int = 10

    # 3. Drawdown protection
    daily_drawdown_limit_pct: float = 5.0  # Pause at -5% daily

    # 4. Win streak sizing
    win_streak_reduce_after: int = 5       # Reduce size after 5 wins
    win_streak_size_reduction: float = 0.5 # Cut size by 50%

    # 5. Correlation filter
    max_correlated_positions: int = 2

    # 6. Time quality filter
    low_quality_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    low_quality_threshold_boost: int = 5   # Add +5 to threshold

    # 7. Max positions
    max_open_positions: int = 5

    # 8. Volatility extreme
    volatility_atr_multiplier: float = 3.0  # Skip if ATR > 3x normal

    # 9. Macro danger (BTC crash protection)
    btc_crash_threshold_pct: float = -5.0   # Pause if BTC -5% in 1h
    btc_crash_pause_minutes: int = 60

CIRCUIT_BREAKERS = CircuitBreakers()


# =============================================================================
# SCANNER CONFIGURATION
# =============================================================================
@dataclass
class ScannerConfig:
    """Market scanner settings."""
    # Pair filtering
    min_volume_usdt_24h: float = 10_000_000  # $10M minimum
    max_pairs_to_analyze: int = 50           # Top 50 by volume
    excluded_pairs: List[str] = field(default_factory=lambda: [
        "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "EURUSDT"  # Stablecoins
    ])

    # Scan timing
    scan_interval_minutes: int = 15          # Every 15min candle close
    scan_offset_seconds: int = 5             # Wait 5s after candle close

    # Rate limiting
    api_calls_per_minute: int = 1200         # Binance limit
    batch_size: int = 10                     # Pairs per batch

SCANNER_CONFIG = ScannerConfig()


# =============================================================================
# TECHNICAL INDICATOR PARAMETERS
# =============================================================================
@dataclass
class IndicatorParams:
    """Technical indicator settings."""
    # RSI
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    # EMA
    ema_fast: int = 9
    ema_medium: int = 21
    ema_slow: int = 50
    ema_trend: int = 200

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # Stochastic RSI
    stoch_rsi_period: int = 14
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3

    # Volume
    volume_ma_period: int = 20
    volume_spike_multiplier: float = 2.0

    # OBV
    obv_ma_period: int = 20

    # VWAP (session-based)
    vwap_std_bands: float = 2.0

    # Support/Resistance
    sr_lookback: int = 100
    sr_tolerance_pct: float = 0.5

    # Patterns
    pattern_min_candles: int = 5

INDICATOR_PARAMS = IndicatorParams()


# =============================================================================
# ORDER FLOW PARAMETERS
# =============================================================================
@dataclass
class OrderFlowParams:
    """Order book and trade flow settings."""
    # Order book depth
    orderbook_depth_levels: int = 20
    imbalance_threshold: float = 0.60       # 60% = significant imbalance

    # Trade aggression
    aggression_window_seconds: int = 300    # 5min window
    aggression_threshold: float = 0.65      # 65% buy/sell = strong

    # Funding rate
    funding_extreme_threshold: float = 0.01  # 1% = extreme

    # Open interest
    oi_change_significant_pct: float = 5.0   # 5% change = significant

ORDERFLOW_PARAMS = OrderFlowParams()


# =============================================================================
# ON-CHAIN PARAMETERS
# =============================================================================
@dataclass
class OnChainParams:
    """On-chain data settings."""
    # Whale alert
    whale_min_usd: float = 1_000_000        # $1M+ = whale
    whale_exchange_flow_window_hours: int = 24

    # Exchange flows
    exchange_inflow_bearish_threshold: float = 10_000_000   # $10M inflow = bearish
    exchange_outflow_bullish_threshold: float = 10_000_000  # $10M outflow = bullish

    # Stablecoin flows
    stablecoin_inflow_bullish_threshold: float = 50_000_000  # $50M = buying pressure

ONCHAIN_PARAMS = OnChainParams()


# =============================================================================
# SENTIMENT PARAMETERS
# =============================================================================
@dataclass
class SentimentParams:
    """Sentiment and macro settings."""
    # Fear & Greed Index
    extreme_fear: int = 20                  # Below = contrarian buy
    extreme_greed: int = 80                 # Above = contrarian sell

    # Social volume (relative change)
    social_spike_threshold: float = 2.0     # 2x normal = significant

    # Market regime
    trending_adx_threshold: float = 25.0    # ADX > 25 = trending
    ranging_adx_threshold: float = 20.0     # ADX < 20 = ranging

SENTIMENT_PARAMS = SentimentParams()


# =============================================================================
# BINANCE API SETTINGS
# =============================================================================
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1"
BINANCE_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds (exponential backoff base)
BATCH_SIZE = 10
BATCH_DELAY = 0.1  # seconds between batches

# Fallback pairs if API fails
FALLBACK_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "NEARUSDT", "FILUSDT", "INJUSDT",
]


# =============================================================================
# API KEYS (from environment)
# =============================================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_TIMEOUT = 60  # seconds

# Optional APIs
WHALE_ALERT_API_KEY = os.getenv("WHALE_ALERT_API_KEY", "")
LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY", "")
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "")


# =============================================================================
# DATABASE
# =============================================================================
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/trading.db")


# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


# =============================================================================
# USER SETTINGS (Runtime modifiable)
# =============================================================================
@dataclass
class UserSettings:
    """User-configurable settings (can be changed via /set command)."""
    account_balance: float = 2500.0         # Default balance
    risk_per_trade_pct: float = 2.0         # User's risk preference
    alerts_enabled: bool = True
    alert_minimum_score: int = 65
    preferred_pairs: List[str] = field(default_factory=list)  # Empty = all

    def calculate_risk_amount(self) -> float:
        """Calculate dollar risk per trade."""
        return self.account_balance * (self.risk_per_trade_pct / 100)

# Default user settings (loaded from DB in production)
DEFAULT_USER_SETTINGS = UserSettings()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def validate_config() -> bool:
    """Validate all configuration parameters."""
    errors = []
    warnings = []

    # Check required API keys
    if not TELEGRAM_TOKEN:
        errors.append("TELEGRAM_TOKEN not set")

    # Binance API key is optional for public data (klines, tickers)
    if not BINANCE_API_KEY:
        warnings.append("BINANCE_API_KEY not set - using public endpoints only")

    # Check weights
    try:
        LayerWeights()
    except ValueError as e:
        errors.append(str(e))

    # Check TP sizes
    try:
        RiskConfig()
    except ValueError as e:
        errors.append(str(e))

    if warnings:
        for warning in warnings:
            print(f"⚠️  Config Warning: {warning}")

    if errors:
        for error in errors:
            print(f"❌ Config Error: {error}")
        return False

    return True


def get_config_summary() -> str:
    """Return a summary of current configuration."""
    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 CONFIGURATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Timeframe: {PRIMARY_TIMEFRAME}
Confirmation: {', '.join(CONFIRMATION_TIMEFRAMES)}

🎯 Weights:
  TA: {LAYER_WEIGHTS.technical*100:.0f}%
  Flow: {LAYER_WEIGHTS.orderflow*100:.0f}%
  Chain: {LAYER_WEIGHTS.onchain*100:.0f}%
  Sentiment: {LAYER_WEIGHTS.sentiment*100:.0f}%
  Backtest: {LAYER_WEIGHTS.backtest*100:.0f}%

⚖️ Risk:
  Per Trade: {RISK_CONFIG.risk_per_trade_pct}%
  Max Heat: {RISK_CONFIG.max_portfolio_heat}%
  Min R:R: {RISK_CONFIG.min_rr_ratio}

💰 Take Profits:
  TP1: {RISK_CONFIG.tp1_rr}R → {RISK_CONFIG.tp1_size*100:.0f}%
  TP2: {RISK_CONFIG.tp2_rr}R → {RISK_CONFIG.tp2_size*100:.0f}%
  TP3: {RISK_CONFIG.tp3_rr}R → {RISK_CONFIG.tp3_size*100:.0f}%

🔒 Circuit Breakers: 9 active
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

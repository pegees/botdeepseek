"""
Strategy Parameters
====================
All tunable strategy parameters for the scalping engine.
"""

# =============================================================================
# SCALP-SPECIFIC SETTINGS (15m focused)
# =============================================================================

# Target hold time
SCALP_TARGET_HOLD_TIME_CANDLES = 4      # Target: hold 1-4 candles (15-60 min)
SCALP_MAX_HOLD_TIME_CANDLES = 8         # Force exit after 8 candles (2 hours)

# Volatility filters
SCALP_MIN_VOLATILITY_ATR = 0.3          # Skip low-volatility pairs (ATR too small)
SCALP_MAX_VOLATILITY_ATR = 5.0          # Skip extreme volatility (chaos mode)
SCALP_VOLUME_SPIKE_MULTIPLIER = 2.0     # Volume must be 2x avg for entry

# =============================================================================
# INDICATOR THRESHOLDS
# =============================================================================

# RSI thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_EXTREME_OVERSOLD = 20
RSI_EXTREME_OVERBOUGHT = 80

# Stochastic RSI
STOCH_RSI_OVERSOLD = 20
STOCH_RSI_OVERBOUGHT = 80

# MACD
MACD_HISTOGRAM_THRESHOLD = 0           # Positive = bullish momentum

# Volume
VOLUME_SPIKE_THRESHOLD = 2.0           # 2x average = significant
VOLUME_DRY_THRESHOLD = 0.5             # Below 0.5x = skip

# =============================================================================
# PATTERN DETECTION
# =============================================================================

# Breakout detection
BREAKOUT_VOLUME_MULTIPLIER = 2.0       # Volume needed for breakout
BREAKOUT_CLOSE_THRESHOLD = 0.003       # Close must be 0.3% beyond level

# Fakeout detection
FAKEOUT_WICK_RATIO = 0.7               # Wick must be 70%+ of candle range
FAKEOUT_BODY_CLOSE_INSIDE = True       # Body must close inside range

# Fair Value Gap
FVG_MIN_GAP_PCT = 0.2                  # Minimum 0.2% gap
FVG_MAX_AGE_CANDLES = 10               # FVG valid for 10 candles

# EMA Bounce
EMA_BOUNCE_TOLERANCE_PCT = 0.2         # Price within 0.2% of EMA
EMA_BOUNCE_CONFIRM_CANDLES = 2         # Wait 2 candles for confirmation

# =============================================================================
# MULTI-TIMEFRAME CONFLUENCE
# =============================================================================

# MTF agreement multipliers
MTF_ALL_AGREE_MULTIPLIER = 1.5         # All TFs agree = 1.5x boost
MTF_PRIMARY_1H_AGREE = 1.2             # 15m + 1h agree = 1.2x
MTF_PRIMARY_ONLY = 1.0                 # Only 15m signal = 1.0x (baseline)
MTF_1H_DISAGREE = 0.7                  # 1h disagrees = 0.7x penalty
MTF_STRONG_DISAGREE = 0.5              # Strong disagreement = 0.5x

# =============================================================================
# ORDER FLOW THRESHOLDS
# =============================================================================

# Book imbalance
BOOK_IMBALANCE_THRESHOLD = 0.60        # 60% = significant imbalance
BOOK_STRONG_IMBALANCE = 0.75           # 75% = strong signal

# Trade aggression
AGGRESSION_THRESHOLD = 0.65            # 65% buy/sell = directional
AGGRESSION_STRONG = 0.75               # 75% = very strong

# Wall detection
WALL_SIZE_MULTIPLIER = 5.0             # 5x average level = wall

# =============================================================================
# ENTRY REFINEMENT
# =============================================================================

# Entry timing (use 1m/5m for precise entry)
ENTRY_PULLBACK_PCT = 0.15              # Enter on 0.15% pullback from signal
ENTRY_MAX_SLIPPAGE_PCT = 0.05          # Max 0.05% slippage accepted
ENTRY_TIMEOUT_SECONDS = 60             # Cancel entry if not filled in 60s

# =============================================================================
# EXIT MANAGEMENT
# =============================================================================

# Partial exits
TP1_CLOSE_PCT = 40                     # Close 40% at TP1
TP2_CLOSE_PCT = 40                     # Close 40% at TP2
TP3_CLOSE_PCT = 20                     # Close 20% at TP3 (runner)

# Stop loss management
MOVE_SL_TO_BE_AT_TP1 = True            # Move SL to break-even at TP1
TRAIL_SL_TO_TP1_AT_TP2 = True          # Trail SL to TP1 at TP2
ATR_TRAILING_MULTIPLIER = 1.0          # Trail by 1x ATR distance

# =============================================================================
# SIGNAL QUALITY
# =============================================================================

# Minimum conditions for signal
MIN_AGREEING_INDICATORS = 3            # At least 3 indicators must agree
MIN_VOLUME_CONFIRMATION = True         # Volume must confirm direction
REQUIRE_PATTERN_MATCH = False          # Pattern match optional but adds score

# Score adjustments
PATTERN_SCORE_BONUS = 10               # +10 points if pattern matches
MTF_CONFLUENCE_BONUS = 15              # +15 if all timeframes agree
VOLUME_SPIKE_BONUS = 10                # +10 for volume spike
FRESH_BREAKOUT_BONUS = 5               # +5 for breakout < 2 candles old

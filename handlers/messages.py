"""Natural language message handler with improved NLP."""
import re
import logging
from typing import Optional, Tuple

from telegram import Update
from telegram.ext import ContextTypes

from config import VALID_TIMEFRAMES, DEFAULT_TIMEFRAME, FALLBACK_PAIRS
from services.auth import require_auth
from handlers.commands import scalp, multi

logger = logging.getLogger(__name__)

# Intent detection patterns
SIGNAL_PATTERNS = [
    r"\b(scalp|setup|signal|trade|entry|find|scan|check|analyze)\b",
    r"\b(long|short|buy|sell)\b",
    r"\b(what|any|give|show|get)\b.*\b(setup|signal|trade|opportunity|pair)\b",
    r"\b(best|good|top)\b.*\b(trade|setup|signal|pair|entry)\b",
]

# Multi-signal intent patterns
MULTI_PATTERNS = [
    r"\b(top\s*\d+|multiple|several|few|some|list)\b",
    r"\b(best\s*\d+)\b",
    r"\b(give me|show me|find)\b.*\b(\d+|few|some|multiple)\b",
]

# Timeframe detection patterns (more flexible)
TIMEFRAME_PATTERNS = [
    (r"\b1\s*m(?:in(?:ute)?s?)?\b", "1m"),
    (r"\b3\s*m(?:in(?:ute)?s?)?\b", "3m"),
    (r"\b5\s*m(?:in(?:ute)?s?)?\b", "5m"),
    (r"\b15\s*m(?:in(?:ute)?s?)?\b", "15m"),
    (r"\b30\s*m(?:in(?:ute)?s?)?\b", "30m"),
    (r"\b1\s*h(?:our|r)?s?\b", "1h"),
    (r"\b4\s*h(?:our|r)?s?\b", "4h"),
    (r"\bhourly\b", "1h"),
    (r"\b(?:quarter|15)\s*(?:hour|h)\b", "15m"),
    (r"\bhalf\s*(?:hour|h)\b", "30m"),
]

# Specific pair pattern
PAIR_PATTERN = r"\b([A-Z]{2,10})(?:USDT|USD|BUSD)?\b"

# Direction preference patterns
LONG_PATTERNS = [r"\b(long|buy|bull(?:ish)?|up)\b"]
SHORT_PATTERNS = [r"\b(short|sell|bear(?:ish)?|down)\b"]


def detect_intent(text: str) -> bool:
    """Check if the message is a signal request."""
    text_lower = text.lower()
    return any(
        re.search(pattern, text_lower, re.IGNORECASE)
        for pattern in SIGNAL_PATTERNS
    )


def detect_multi_intent(text: str) -> bool:
    """Check if user wants multiple signals."""
    text_lower = text.lower()
    return any(
        re.search(pattern, text_lower, re.IGNORECASE)
        for pattern in MULTI_PATTERNS
    )


def extract_timeframe(text: str) -> str:
    """Extract timeframe from message text."""
    text_lower = text.lower()

    for pattern, timeframe in TIMEFRAME_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return timeframe

    return DEFAULT_TIMEFRAME


def extract_pair(text: str) -> Optional[str]:
    """Extract specific trading pair from message."""
    # Look for pair-like patterns in uppercase
    matches = re.findall(PAIR_PATTERN, text.upper())

    # Check if any match is in our scanned pairs
    for match in matches:
        # Try with USDT suffix
        pair = match if match.endswith("USDT") else f"{match}USDT"
        if pair in FALLBACK_PAIRS:
            return pair

    return None


def extract_direction(text: str) -> Optional[str]:
    """Extract preferred direction (LONG/SHORT) from message."""
    text_lower = text.lower()

    for pattern in LONG_PATTERNS:
        if re.search(pattern, text_lower):
            return "LONG"

    for pattern in SHORT_PATTERNS:
        if re.search(pattern, text_lower):
            return "SHORT"

    return None


@require_auth
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle natural language messages with improved NLP."""
    if not update.message or not update.message.text:
        return

    text = update.message.text
    text_lower = text.lower()

    # Check if this is a signal request
    if not detect_intent(text):
        # Not a signal request - show help
        await update.message.reply_text(
            "🤔 I'm not sure what you need.\n\n"
            "*Try saying:*\n"
            "• \"give me a scalp setup\"\n"
            "• \"find me a long on 5m\"\n"
            "• \"any good shorts?\"\n"
            "• \"top 3 setups\"\n\n"
            "Or use /help for all commands.",
            parse_mode="Markdown",
        )
        return

    # Extract parameters from message
    timeframe = extract_timeframe(text)
    specific_pair = extract_pair(text)
    direction = extract_direction(text)

    # Store preferences in context for potential use
    context.user_data["preferred_direction"] = direction
    context.user_data["specific_pair"] = specific_pair

    # Log what we detected
    logger.info(
        f"NLP detected: timeframe={timeframe}, pair={specific_pair}, "
        f"direction={direction}, multi={detect_multi_intent(text)}"
    )

    # Set args for the handler
    context.args = [timeframe]

    # Determine which handler to call
    if detect_multi_intent(text):
        await multi(update, context)
    else:
        await scalp(update, context)

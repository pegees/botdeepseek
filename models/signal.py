"""Signal data model with parsing logic."""
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Represents a trading signal with entry, stop loss, and take profit."""

    pair: str
    direction: str  # "LONG" or "SHORT"
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: str
    confidence: str  # "LOW", "MEDIUM", or "HIGH"
    reasoning: str
    timeframe: str = "15m"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[int] = None

    @property
    def sl_percent(self) -> str:
        """Calculate stop loss percentage from entry."""
        if self.entry == 0:
            return "0.00"
        pct = abs((self.stop_loss - self.entry) / self.entry * 100)
        return f"{pct:.2f}"

    @property
    def tp_percent(self) -> str:
        """Calculate take profit percentage from entry."""
        if self.entry == 0:
            return "0.00"
        pct = abs((self.take_profit - self.entry) / self.entry * 100)
        return f"{pct:.2f}"

    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.direction.upper() == "LONG"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pair": self.pair,
            "direction": self.direction,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward": self.risk_reward,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Create Signal from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        return cls(
            pair=data["pair"],
            direction=data["direction"],
            entry=float(data["entry"]),
            stop_loss=float(data["stop_loss"]),
            take_profit=float(data["take_profit"]),
            risk_reward=data.get("risk_reward", "N/A"),
            confidence=data.get("confidence", "MEDIUM"),
            reasoning=data.get("reasoning", ""),
            timeframe=data.get("timeframe", "15m"),
            timestamp=timestamp,
            user_id=data.get("user_id"),
        )

    @classmethod
    def from_deepseek_response(
        cls, response: str, timeframe: str = "15m", user_id: Optional[int] = None
    ) -> Optional["Signal"]:
        """
        Parse a DeepSeek response into a Signal object.

        Args:
            response: Raw response text from DeepSeek
            timeframe: Trading timeframe
            user_id: Optional user ID who requested the signal

        Returns:
            Signal object if parsing succeeds, None if no setup found
        """
        if not response:
            return None

        # Check for no setup response
        if "NO CLEAR SETUP" in response.upper():
            logger.info("DeepSeek returned no clear setup")
            return None

        try:
            # Extract fields using regex patterns
            pair_match = re.search(r"PAIR:\s*([A-Z]+(?:USDT|USD|BUSD)?)", response, re.IGNORECASE)
            direction_match = re.search(r"DIRECTION:\s*(LONG|SHORT)", response, re.IGNORECASE)
            entry_match = re.search(r"ENTRY:\s*\$?([\d.,]+)", response, re.IGNORECASE)
            sl_match = re.search(r"STOP\s*LOSS:\s*\$?([\d.,]+)", response, re.IGNORECASE)
            tp_match = re.search(r"TAKE\s*PROFIT:\s*\$?([\d.,]+)", response, re.IGNORECASE)
            rr_match = re.search(r"RISK/?REWARD:\s*([\d.:]+)", response, re.IGNORECASE)
            conf_match = re.search(r"CONFIDENCE:\s*(LOW|MEDIUM|HIGH)", response, re.IGNORECASE)
            reason_match = re.search(r"REASONING:\s*(.+?)(?:\n\n|$)", response, re.IGNORECASE | re.DOTALL)

            # Validate required fields
            if not all([pair_match, direction_match, entry_match, sl_match, tp_match]):
                logger.warning(f"Failed to parse required fields from response: {response[:200]}")
                return None

            # Parse numeric values
            def parse_number(match) -> float:
                if not match:
                    return 0.0
                value = match.group(1).replace(",", "")
                return float(value)

            pair = pair_match.group(1).upper()
            if not pair.endswith(("USDT", "USD", "BUSD")):
                pair = pair + "USDT"

            return cls(
                pair=pair,
                direction=direction_match.group(1).upper(),
                entry=parse_number(entry_match),
                stop_loss=parse_number(sl_match),
                take_profit=parse_number(tp_match),
                risk_reward=rr_match.group(1) if rr_match else "N/A",
                confidence=conf_match.group(1).upper() if conf_match else "MEDIUM",
                reasoning=reason_match.group(1).strip() if reason_match else "",
                timeframe=timeframe,
                user_id=user_id,
            )

        except Exception as e:
            logger.error(f"Error parsing DeepSeek response: {e}")
            logger.debug(f"Response was: {response}")
            return None

    @classmethod
    def parse_multi_response(
        cls, response: str, timeframe: str = "15m", user_id: Optional[int] = None
    ) -> List["Signal"]:
        """
        Parse a multi-signal DeepSeek response.

        Args:
            response: Raw response text containing multiple setups
            timeframe: Trading timeframe
            user_id: Optional user ID who requested the signals

        Returns:
            List of Signal objects (may be empty)
        """
        if not response or "NO CLEAR SETUP" in response.upper():
            return []

        signals = []

        # Split by "SETUP" markers
        setup_pattern = r"SETUP\s*\d+:?\s*"
        parts = re.split(setup_pattern, response, flags=re.IGNORECASE)

        for part in parts:
            if not part.strip():
                continue

            signal = cls.from_deepseek_response(part, timeframe, user_id)
            if signal:
                signals.append(signal)

        # If no SETUP markers found, try parsing as single response
        if not signals:
            single = cls.from_deepseek_response(response, timeframe, user_id)
            if single:
                signals.append(single)

        logger.info(f"Parsed {len(signals)} signals from multi-response")
        return signals

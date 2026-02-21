"""
Structured Logging System
==========================
Centralized logging with structured output for debugging and audit trails.
"""
import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path
import json

from config import LOG_LEVEL, LOG_FORMAT


class StructuredLogger:
    """
    Structured logger that outputs JSON-formatted logs for easy parsing.
    Every action is logged for complete audit trail.
    """

    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOG_LEVEL))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        self.logger.addHandler(console_handler)

        # File handler (daily rotation)
        log_file = self.log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        self.logger.addHandler(file_handler)

    def _format_structured(self, level: str, message: str, **kwargs) -> Dict:
        """Format log entry as structured JSON."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
        }
        entry.update(kwargs)
        return entry

    def info(self, message: str, **kwargs):
        """Log info level message."""
        self.logger.info(message)
        if kwargs:
            structured = self._format_structured("INFO", message, **kwargs)
            self._write_structured(structured)

    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self.logger.warning(message)
        structured = self._format_structured("WARNING", message, **kwargs)
        self._write_structured(structured)

    def error(self, message: str, **kwargs):
        """Log error level message."""
        self.logger.error(message)
        structured = self._format_structured("ERROR", message, **kwargs)
        self._write_structured(structured)

    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self.logger.debug(message)

    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self.logger.critical(message)
        structured = self._format_structured("CRITICAL", message, **kwargs)
        self._write_structured(structured)

    def _write_structured(self, entry: Dict):
        """Write structured entry to JSON log file."""
        json_file = self.log_dir / f"{datetime.now().strftime('%Y-%m-%d')}_structured.jsonl"
        with open(json_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # =========================================================================
    # Domain-Specific Logging Methods
    # =========================================================================

    def log_signal(self, signal: Dict):
        """Log a generated signal."""
        self.info(
            f"SIGNAL: {signal.get('pair')} {signal.get('direction')} "
            f"score={signal.get('confluence_score')}",
            event_type="signal_generated",
            pair=signal.get("pair"),
            direction=signal.get("direction"),
            score=signal.get("confluence_score"),
            tier=signal.get("signal_tier"),
            entry=signal.get("entry_price"),
            sl=signal.get("stop_loss"),
            tp1=signal.get("tp1"),
        )

    def log_signal_discarded(self, pair: str, reason: str, score: float = None):
        """Log a discarded signal."""
        self.debug(
            f"SIGNAL DISCARDED: {pair} - {reason}",
            event_type="signal_discarded",
            pair=pair,
            reason=reason,
            score=score,
        )

    def log_circuit_breaker(self, breaker_type: str, reason: str, pair: str = None):
        """Log circuit breaker activation."""
        self.warning(
            f"CIRCUIT BREAKER: {breaker_type} - {reason}",
            event_type="circuit_breaker",
            breaker_type=breaker_type,
            reason=reason,
            pair=pair,
        )

    def log_trade_outcome(self, pair: str, direction: str, outcome: str, pnl_pct: float):
        """Log trade outcome."""
        self.info(
            f"TRADE CLOSED: {pair} {direction} → {outcome} ({pnl_pct:+.2f}%)",
            event_type="trade_closed",
            pair=pair,
            direction=direction,
            outcome=outcome,
            pnl_pct=pnl_pct,
        )

    def log_whale_event(self, token: str, amount_usd: float, direction: str):
        """Log whale movement."""
        self.info(
            f"WHALE: ${amount_usd/1e6:.1f}M {token} {direction}",
            event_type="whale_alert",
            token=token,
            amount_usd=amount_usd,
            direction=direction,
        )

    def log_scan(self, pairs_scanned: int, qualified: int, signals: int, duration_ms: float):
        """Log market scan."""
        self.info(
            f"SCAN: {pairs_scanned} pairs → {qualified} qualified → {signals} signals ({duration_ms:.0f}ms)",
            event_type="market_scan",
            pairs_scanned=pairs_scanned,
            qualified=qualified,
            signals=signals,
            duration_ms=duration_ms,
        )

    def log_api_error(self, api: str, error: str, endpoint: str = None):
        """Log API error."""
        self.error(
            f"API ERROR: {api} - {error}",
            event_type="api_error",
            api=api,
            error=error,
            endpoint=endpoint,
        )

    def log_ws_event(self, event: str, details: str = None):
        """Log WebSocket event."""
        self.info(
            f"WEBSOCKET: {event}",
            event_type="websocket",
            event=event,
            details=details,
        )


# Singleton loggers
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str = "scalp_engine") -> StructuredLogger:
    """Get or create a logger instance."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


# Convenience functions
def log_signal(signal: Dict):
    get_logger().log_signal(signal)


def log_circuit_breaker(breaker_type: str, reason: str, pair: str = None):
    get_logger().log_circuit_breaker(breaker_type, reason, pair)


def log_trade_outcome(pair: str, direction: str, outcome: str, pnl_pct: float):
    get_logger().log_trade_outcome(pair, direction, outcome, pnl_pct)

"""Core modules for API clients, scanner, engine, and exceptions."""
from .exceptions import BotException, BinanceAPIError, DeepSeekAPIError, RateLimitError, AuthenticationError, DataFetchError, SignalParseError
from .binance import BinanceClient
from .deepseek import DeepSeekClient
from .scanner import MarketScanner, PairInfo, ScanResult, get_scanner, scan_market
from .confluence import ConfluenceEngine, ConfluenceResult, get_confluence_engine, calculate_confluence
from .engine import TradingEngine, SignalOutput, get_engine, run_scan

__all__ = [
    "BotException",
    "BinanceAPIError",
    "DeepSeekAPIError",
    "RateLimitError",
    "AuthenticationError",
    "DataFetchError",
    "SignalParseError",
    "BinanceClient",
    "DeepSeekClient",
    "MarketScanner",
    "PairInfo",
    "ScanResult",
    "get_scanner",
    "scan_market",
    "ConfluenceEngine",
    "ConfluenceResult",
    "get_confluence_engine",
    "calculate_confluence",
    "TradingEngine",
    "SignalOutput",
    "get_engine",
    "run_scan",
]

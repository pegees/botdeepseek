"""Custom exceptions for the Scalp Signal Bot."""


class BotException(Exception):
    """Base exception for all bot errors."""
    pass


class BinanceAPIError(BotException):
    """Raised when a Binance API call fails."""
    pass


class DeepSeekAPIError(BotException):
    """Raised when a DeepSeek API call fails."""
    pass


class RateLimitError(BotException):
    """Raised when an API rate limit is exceeded."""
    pass


class AuthenticationError(BotException):
    """Raised when a user is not authorized to use the bot."""
    pass


class DataFetchError(BotException):
    """Raised when market data cannot be fetched."""
    pass


class SignalParseError(BotException):
    """Raised when a signal response cannot be parsed."""
    pass

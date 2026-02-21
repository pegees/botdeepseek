"""User authentication and whitelist management."""
import logging
from functools import wraps
from typing import Callable, Any

from telegram import Update
from telegram.ext import ContextTypes

from config import ALLOWED_USER_IDS, WHITELIST_ENABLED

logger = logging.getLogger(__name__)


def is_authorized(user_id: int) -> bool:
    """
    Check if a user ID is authorized to use the bot.

    Args:
        user_id: Telegram user ID to check

    Returns:
        True if user is authorized, False otherwise
    """
    if not WHITELIST_ENABLED:
        return True
    return user_id in ALLOWED_USER_IDS


def require_auth(func: Callable) -> Callable:
    """
    Decorator to check if user is whitelisted before executing handler.

    Usage:
        @require_auth
        async def my_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            ...
    """
    @wraps(func)
    async def wrapper(
        update: Update, context: ContextTypes.DEFAULT_TYPE, *args: Any, **kwargs: Any
    ) -> Any:
        # Skip auth check if whitelist is disabled
        if not WHITELIST_ENABLED:
            return await func(update, context, *args, **kwargs)

        # Get user info
        user = update.effective_user
        if not user:
            logger.warning("Received update without user information")
            return

        user_id = user.id
        username = user.username or "unknown"

        # Check authorization
        if user_id not in ALLOWED_USER_IDS:
            logger.warning(
                f"Unauthorized access attempt: user_id={user_id}, username=@{username}"
            )
            if update.message:
                await update.message.reply_text(
                    "You are not authorized to use this bot.\n\n"
                    "Contact the administrator to request access.\n"
                    f"Your user ID: `{user_id}`",
                    parse_mode="Markdown",
                )
            return

        # User is authorized, proceed with handler
        logger.debug(f"Authorized request from user_id={user_id}, username=@{username}")
        return await func(update, context, *args, **kwargs)

    return wrapper


def add_user(user_id: int) -> bool:
    """
    Add a user to the whitelist (runtime only, not persisted).

    Args:
        user_id: Telegram user ID to add

    Returns:
        True if user was added, False if already exists
    """
    if user_id in ALLOWED_USER_IDS:
        return False
    ALLOWED_USER_IDS.add(user_id)
    logger.info(f"Added user {user_id} to whitelist")
    return True


def remove_user(user_id: int) -> bool:
    """
    Remove a user from the whitelist (runtime only, not persisted).

    Args:
        user_id: Telegram user ID to remove

    Returns:
        True if user was removed, False if not found
    """
    if user_id not in ALLOWED_USER_IDS:
        return False
    ALLOWED_USER_IDS.discard(user_id)
    logger.info(f"Removed user {user_id} from whitelist")
    return True

#!/usr/bin/env python3
"""
Scalp Signal Bot - Main Entry Point

A Telegram bot that scans Binance Futures markets and uses DeepSeek AI
to find scalp trading setups, enhanced with multi-indicator analysis.

Usage:
    export TELEGRAM_TOKEN="your_token"
    export DEEPSEEK_API_KEY="your_key"
    export BINANCE_API_KEY="your_key"
    export BINANCE_API_SECRET="your_secret"
    python bot.py
"""
import logging
import sys

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config import (
    TELEGRAM_TOKEN, setup_logging, validate_config,
    DASHBOARD_ENABLED, DASHBOARD_HOST, DASHBOARD_PORT
)
from handlers import (
    start,
    help_command,
    scalp,
    multi,
    history,
    status,
    pairs,
    alert,
    scan,
    dashboard,
    handle_message,
)
from services.scheduler import init_scheduler

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


async def post_init(application: Application) -> None:
    """Initialize services after application starts."""
    # Initialize and start the alert scheduler
    scheduler = init_scheduler(application)
    await scheduler.start()
    logger.info("Post-init complete: scheduler started")

    # Start dashboard if enabled
    if DASHBOARD_ENABLED:
        try:
            from dashboard import run_dashboard
            run_dashboard(host=DASHBOARD_HOST, port=DASHBOARD_PORT)
            logger.info(f"Dashboard available at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
        except ImportError:
            logger.warning("Dashboard dependencies not installed (fastapi, uvicorn). Skipping dashboard.")
        except Exception as e:
            logger.warning(f"Failed to start dashboard: {e}")


async def post_shutdown(application: Application) -> None:
    """Cleanup on shutdown."""
    from services.scheduler import get_scheduler

    scheduler = get_scheduler()
    if scheduler:
        scheduler.stop()
    logger.info("Shutdown complete")


async def error_handler(update: object, context) -> None:
    """Handle errors in the bot."""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

    # Send message to user if possible
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "An error occurred while processing your request.\n"
                "Please try again or contact the administrator."
            )
        except Exception:
            pass


def main() -> None:
    """Start the bot."""
    # Validate configuration before starting
    validate_config()

    logger.info("=" * 50)
    logger.info("Starting Scalp Signal Bot...")
    logger.info("Multi-indicator scanner enabled")
    logger.info("=" * 50)

    # Build application
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # Register command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("scalp", scalp))
    app.add_handler(CommandHandler("scan", scan))
    app.add_handler(CommandHandler("multi", multi))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("pairs", pairs))
    app.add_handler(CommandHandler("alert", alert))
    app.add_handler(CommandHandler("dashboard", dashboard))

    # Register message handler for natural language
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Register error handler
    app.add_error_handler(error_handler)

    logger.info("Bot is running! Send /start to your bot on Telegram.")
    logger.info("Press Ctrl+C to stop.")

    # Start polling
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

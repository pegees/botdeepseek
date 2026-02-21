"""Signal history storage using SQLite."""
import sqlite3
import logging
import asyncio
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from functools import partial

from config import DATABASE_PATH
from models.signal import Signal

logger = logging.getLogger(__name__)


class HistoryService:
    """Manages signal history storage and retrieval using SQLite."""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                risk_reward TEXT,
                confidence TEXT,
                reasoning TEXT,
                timeframe TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER
            )
        """)

        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp
            ON signals(timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_user
            ON signals(user_id, timestamp DESC)
        """)

        # Create user stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id INTEGER PRIMARY KEY,
                total_requests INTEGER DEFAULT 0,
                last_request DATETIME,
                alerts_enabled INTEGER DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def add_signal(self, signal: Signal) -> int:
        """
        Store a signal in the database.

        Args:
            signal: Signal object to store

        Returns:
            ID of the inserted signal
        """
        def _insert() -> int:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO signals (
                    pair, direction, entry, stop_loss, take_profit,
                    risk_reward, confidence, reasoning, timeframe,
                    timestamp, user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.pair,
                signal.direction,
                signal.entry,
                signal.stop_loss,
                signal.take_profit,
                signal.risk_reward,
                signal.confidence,
                signal.reasoning,
                signal.timeframe,
                signal.timestamp.isoformat(),
                signal.user_id,
            ))

            signal_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return signal_id

        loop = asyncio.get_event_loop()
        signal_id = await loop.run_in_executor(None, _insert)
        logger.info(f"Stored signal {signal_id}: {signal.pair} {signal.direction}")
        return signal_id

    async def get_recent(
        self, limit: int = 5, user_id: Optional[int] = None
    ) -> List[Signal]:
        """
        Get recent signals.

        Args:
            limit: Maximum number of signals to return
            user_id: Filter by user ID (None for all users)

        Returns:
            List of Signal objects, most recent first
        """
        def _query() -> List[Dict]:
            conn = self._get_connection()
            cursor = conn.cursor()

            if user_id is not None:
                cursor.execute("""
                    SELECT pair, direction, entry, stop_loss, take_profit,
                           risk_reward, confidence, reasoning, timeframe,
                           timestamp, user_id
                    FROM signals
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, limit))
            else:
                cursor.execute("""
                    SELECT pair, direction, entry, stop_loss, take_profit,
                           risk_reward, confidence, reasoning, timeframe,
                           timestamp, user_id
                    FROM signals
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows

        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, _query)

        signals = []
        for row in rows:
            try:
                signals.append(Signal.from_dict(row))
            except Exception as e:
                logger.error(f"Error parsing signal from DB: {e}")

        return signals

    async def get_stats(self) -> Dict[str, int]:
        """
        Get signal statistics.

        Returns:
            Dict with 'today' and 'total' signal counts
        """
        def _query() -> Dict[str, int]:
            conn = self._get_connection()
            cursor = conn.cursor()

            today = date.today().isoformat()

            cursor.execute("""
                SELECT COUNT(*) FROM signals
                WHERE DATE(timestamp) = ?
            """, (today,))
            today_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM signals")
            total_count = cursor.fetchone()[0]

            conn.close()
            return {"today": today_count, "total": total_count}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _query)

    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Get stats for a specific user.

        Args:
            user_id: Telegram user ID

        Returns:
            Dict with user statistics
        """
        def _query() -> Dict[str, Any]:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get or create user stats
            cursor.execute("""
                SELECT total_requests, last_request, alerts_enabled
                FROM user_stats WHERE user_id = ?
            """, (user_id,))
            row = cursor.fetchone()

            if row:
                stats = {
                    "total_requests": row["total_requests"],
                    "last_request": row["last_request"],
                    "alerts_enabled": bool(row["alerts_enabled"]),
                }
            else:
                stats = {
                    "total_requests": 0,
                    "last_request": None,
                    "alerts_enabled": False,
                }

            # Get today's request count
            today = date.today().isoformat()
            cursor.execute("""
                SELECT COUNT(*) FROM signals
                WHERE user_id = ? AND DATE(timestamp) = ?
            """, (user_id, today))
            stats["requests_today"] = cursor.fetchone()[0]

            conn.close()
            return stats

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _query)

    async def increment_user_requests(self, user_id: int) -> None:
        """
        Increment user request count and update last request time.

        Args:
            user_id: Telegram user ID
        """
        def _update():
            conn = self._get_connection()
            cursor = conn.cursor()

            now = datetime.utcnow().isoformat()

            cursor.execute("""
                INSERT INTO user_stats (user_id, total_requests, last_request)
                VALUES (?, 1, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    total_requests = total_requests + 1,
                    last_request = ?
            """, (user_id, now, now))

            conn.commit()
            conn.close()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _update)

    async def set_alerts_enabled(self, user_id: int, enabled: bool) -> None:
        """
        Set alert preference for a user.

        Args:
            user_id: Telegram user ID
            enabled: Whether alerts should be enabled
        """
        def _update():
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO user_stats (user_id, alerts_enabled)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    alerts_enabled = ?
            """, (user_id, int(enabled), int(enabled)))

            conn.commit()
            conn.close()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _update)

    async def get_users_with_alerts(self) -> List[int]:
        """
        Get all user IDs with alerts enabled.

        Returns:
            List of user IDs
        """
        def _query() -> List[int]:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT user_id FROM user_stats
                WHERE alerts_enabled = 1
            """)
            user_ids = [row["user_id"] for row in cursor.fetchall()]

            conn.close()
            return user_ids

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _query)


# Singleton instance
history_service = HistoryService()

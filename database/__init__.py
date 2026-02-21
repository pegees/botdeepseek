"""Database module for storing signals, positions, and performance."""
import os
import sqlite3
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


class Database:
    """SQLite database wrapper for the trading bot."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_PATH

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        """Initialize database schema."""
        schema_path = Path(__file__).parent / "schema.sql"

        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema = f.read()

            conn = self._get_conn()
            conn.executescript(schema)
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        else:
            logger.warning(f"Schema file not found at {schema_path}")

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # Signal operations
    def save_signal(self, signal_data: Dict) -> int:
        """Save a new signal and return its ID."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signals (
                symbol, direction, timeframe, entry_price, stop_loss,
                tp1_price, tp2_price, tp3_price, confluence_score,
                ta_score, orderflow_score, onchain_score, sentiment_score,
                position_size_usd, risk_amount_usd, rr_ratio, atr, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_data['symbol'],
            signal_data['direction'],
            signal_data.get('timeframe', '15m'),
            signal_data['entry_price'],
            signal_data['stop_loss'],
            signal_data['tp1_price'],
            signal_data['tp2_price'],
            signal_data['tp3_price'],
            signal_data['confluence_score'],
            signal_data.get('ta_score'),
            signal_data.get('orderflow_score'),
            signal_data.get('onchain_score'),
            signal_data.get('sentiment_score'),
            signal_data.get('position_size_usd'),
            signal_data.get('risk_amount_usd'),
            signal_data.get('rr_ratio'),
            signal_data.get('atr'),
            signal_data.get('expires_at')
        ))

        conn.commit()
        return cursor.lastrowid

    def update_signal_status(self, signal_id: int, status: str):
        """Update signal status."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE signals SET status = ? WHERE id = ?",
            (status, signal_id)
        )
        conn.commit()

    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Get recent signals."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    # Position operations
    def open_position(self, position_data: Dict) -> int:
        """Open a new position."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO positions (
                signal_id, symbol, direction, entry_price, entry_time, position_size
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            position_data['signal_id'],
            position_data['symbol'],
            position_data['direction'],
            position_data['entry_price'],
            position_data.get('entry_time', datetime.now()),
            position_data['position_size']
        ))

        conn.commit()
        return cursor.lastrowid

    def close_position(self, position_id: int, exit_data: Dict):
        """Close a position."""
        conn = self._get_conn()
        conn.execute("""
            UPDATE positions SET
                exit_price = ?,
                exit_time = ?,
                exit_reason = ?,
                realized_pnl = ?,
                realized_pnl_pct = ?,
                status = 'closed'
            WHERE id = ?
        """, (
            exit_data['exit_price'],
            exit_data.get('exit_time', datetime.now()),
            exit_data['exit_reason'],
            exit_data['realized_pnl'],
            exit_data['realized_pnl_pct'],
            position_id
        ))
        conn.commit()

    def get_active_positions(self) -> List[Dict]:
        """Get all active positions."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM v_active_positions"
        )
        return [dict(row) for row in cursor.fetchall()]

    # Performance operations
    def get_today_stats(self) -> Dict:
        """Get today's performance stats."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM v_today_performance")
        row = cursor.fetchone()
        return dict(row) if row else {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl_pct': 0,
            'avg_pnl_pct': 0
        }

    def log_scan(self, scan_data: Dict):
        """Log a market scan."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO scan_logs (
                pairs_scanned, pairs_qualified, signals_generated,
                scan_duration_ms, top_pair, top_score
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            scan_data['pairs_scanned'],
            scan_data['pairs_qualified'],
            scan_data['signals_generated'],
            scan_data['scan_duration_ms'],
            scan_data.get('top_pair'),
            scan_data.get('top_score')
        ))
        conn.commit()

    # User settings
    def get_user_settings(self, user_id: str) -> Dict:
        """Get user settings."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM user_settings WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else {
            'account_balance': 2500.0,
            'risk_per_trade_pct': 2.0,
            'alerts_enabled': True,
            'alert_minimum_score': 65
        }

    def save_user_settings(self, user_id: str, settings: Dict):
        """Save user settings."""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO user_settings (
                user_id, account_balance, risk_per_trade_pct,
                alerts_enabled, alert_minimum_score, preferred_pairs
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            settings.get('account_balance', 2500.0),
            settings.get('risk_per_trade_pct', 2.0),
            settings.get('alerts_enabled', True),
            settings.get('alert_minimum_score', 65),
            settings.get('preferred_pairs')
        ))
        conn.commit()


# Singleton
_db: Optional[Database] = None


def get_database() -> Database:
    """Get database singleton."""
    global _db
    if _db is None:
        _db = Database()
    return _db

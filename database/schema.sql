-- ============================================================================
-- SCALP BOT DATABASE SCHEMA
-- Professional-grade 15m crypto scalping system
-- ============================================================================

-- Signals table: Store all generated signals
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    timeframe TEXT NOT NULL DEFAULT '15m',

    -- Entry/Exit levels
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    tp1_price REAL NOT NULL,
    tp2_price REAL NOT NULL,
    tp3_price REAL NOT NULL,

    -- Scoring
    confluence_score REAL NOT NULL,
    ta_score REAL,
    orderflow_score REAL,
    onchain_score REAL,
    sentiment_score REAL,
    backtest_score REAL,

    -- Risk
    position_size_usd REAL,
    risk_amount_usd REAL,
    rr_ratio REAL,
    atr REAL,

    -- Status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'tp1_hit', 'tp2_hit', 'tp3_hit', 'sl_hit', 'expired', 'cancelled')),
    expires_at TIMESTAMP,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Positions table: Track active and closed positions
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER REFERENCES signals(id),
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,

    -- Entry
    entry_price REAL NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    position_size REAL NOT NULL,

    -- Exit
    exit_price REAL,
    exit_time TIMESTAMP,
    exit_reason TEXT CHECK (exit_reason IN ('tp1', 'tp2', 'tp3', 'sl', 'manual', 'expired')),

    -- Partial closes
    tp1_filled BOOLEAN DEFAULT FALSE,
    tp1_fill_time TIMESTAMP,
    tp2_filled BOOLEAN DEFAULT FALSE,
    tp2_fill_time TIMESTAMP,
    tp3_filled BOOLEAN DEFAULT FALSE,
    tp3_fill_time TIMESTAMP,

    -- P&L
    realized_pnl REAL,
    realized_pnl_pct REAL,

    -- Status
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'partial', 'closed')),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance table: Daily/weekly/monthly stats
CREATE TABLE IF NOT EXISTS performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period_type TEXT NOT NULL CHECK (period_type IN ('daily', 'weekly', 'monthly')),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Trade stats
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate REAL,

    -- P&L
    gross_profit REAL DEFAULT 0,
    gross_loss REAL DEFAULT 0,
    net_pnl REAL DEFAULT 0,
    net_pnl_pct REAL DEFAULT 0,

    -- Risk stats
    max_drawdown REAL,
    max_drawdown_pct REAL,
    avg_win REAL,
    avg_loss REAL,
    profit_factor REAL,

    -- Signal stats
    signals_generated INTEGER DEFAULT 0,
    signals_taken INTEGER DEFAULT 0,
    avg_confluence_score REAL,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(period_type, period_start)
);

-- User settings table
CREATE TABLE IF NOT EXISTS user_settings (
    user_id TEXT PRIMARY KEY,
    account_balance REAL DEFAULT 2500.0,
    risk_per_trade_pct REAL DEFAULT 2.0,
    alerts_enabled BOOLEAN DEFAULT TRUE,
    alert_minimum_score INTEGER DEFAULT 65,
    preferred_pairs TEXT,  -- JSON array
    timezone TEXT DEFAULT 'UTC',

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Circuit breaker events
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    breaker_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    signal_id INTEGER REFERENCES signals(id)
);

-- Market scans log
CREATE TABLE IF NOT EXISTS scan_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pairs_scanned INTEGER,
    pairs_qualified INTEGER,
    signals_generated INTEGER,
    scan_duration_ms REAL,
    top_pair TEXT,
    top_score REAL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_performance_period ON performance(period_type, period_start);

-- Views for quick stats

-- Today's performance view
CREATE VIEW IF NOT EXISTS v_today_performance AS
SELECT
    COUNT(*) as total_trades,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_rate,
    ROUND(SUM(realized_pnl_pct), 2) as total_pnl_pct,
    ROUND(AVG(realized_pnl_pct), 2) as avg_pnl_pct
FROM positions
WHERE DATE(created_at) = DATE('now')
AND status = 'closed';

-- Active positions view
CREATE VIEW IF NOT EXISTS v_active_positions AS
SELECT
    p.id,
    p.symbol,
    p.direction,
    p.entry_price,
    p.position_size,
    s.stop_loss,
    s.tp1_price,
    s.tp2_price,
    s.tp3_price,
    s.confluence_score,
    p.tp1_filled,
    p.tp2_filled,
    p.created_at
FROM positions p
JOIN signals s ON p.signal_id = s.id
WHERE p.status IN ('open', 'partial');

-- Recent signals view
CREATE VIEW IF NOT EXISTS v_recent_signals AS
SELECT
    id,
    symbol,
    direction,
    entry_price,
    stop_loss,
    tp1_price,
    confluence_score,
    status,
    created_at
FROM signals
WHERE created_at >= datetime('now', '-24 hours')
ORDER BY created_at DESC
LIMIT 50;

-- Triggers for updated_at

CREATE TRIGGER IF NOT EXISTS update_signals_timestamp
AFTER UPDATE ON signals
BEGIN
    UPDATE signals SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_positions_timestamp
AFTER UPDATE ON positions
BEGIN
    UPDATE positions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_performance_timestamp
AFTER UPDATE ON performance
BEGIN
    UPDATE performance SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_user_settings_timestamp
AFTER UPDATE ON user_settings
BEGIN
    UPDATE user_settings SET updated_at = CURRENT_TIMESTAMP WHERE user_id = NEW.user_id;
END;

"""FastAPI dashboard for real-time signal display."""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import threading

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from services.scanner import ScannerService, ScanResult

logger = logging.getLogger(__name__)

# Global state for dashboard
_scan_results: List[ScanResult] = []
_last_scan_time: Optional[str] = None
_connected_websockets: List[Any] = []


def create_app() -> "FastAPI":
    """Create and configure the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn websockets")

    app = FastAPI(title="Scalp Signal Dashboard", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the dashboard HTML."""
        return get_dashboard_html()

    @app.get("/api/signals")
    async def get_signals():
        """Get current scan results."""
        return {
            "signals": [_result_to_dict(r) for r in _scan_results],
            "last_scan": _last_scan_time,
            "count": len(_scan_results),
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time updates."""
        await websocket.accept()
        _connected_websockets.append(websocket)

        try:
            # Send current state immediately
            await websocket.send_json({
                "type": "initial",
                "signals": [_result_to_dict(r) for r in _scan_results],
                "last_scan": _last_scan_time,
            })

            # Keep connection open
            while True:
                try:
                    # Wait for messages (ping/pong)
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    # Send ping to keep alive
                    await websocket.send_text("ping")

        except WebSocketDisconnect:
            _connected_websockets.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket in _connected_websockets:
                _connected_websockets.remove(websocket)

    @app.post("/api/scan")
    async def trigger_scan(timeframe: str = "15m"):
        """Manually trigger a scan."""
        scanner = ScannerService()
        results = await scanner.scan(timeframe=timeframe)
        await update_results(results)
        return {"status": "ok", "count": len(results)}

    return app


def _result_to_dict(result: ScanResult) -> Dict[str, Any]:
    """Convert ScanResult to JSON-serializable dict."""
    return {
        "symbol": result.symbol,
        "price": result.price,
        "signal": result.signal,
        "confidence": result.confidence,
        "score": result.score,
        "indicators": result.indicators,
        "confluence": result.confluence,
        "timeframe": result.timeframe,
        "metadata": result.metadata,
    }


async def update_results(results: List[ScanResult], scan_time: Optional[str] = None) -> None:
    """Update global results and notify all connected clients."""
    global _scan_results, _last_scan_time

    _scan_results = results
    _last_scan_time = scan_time or asyncio.get_event_loop().time()

    # Notify all connected WebSocket clients
    message = {
        "type": "update",
        "signals": [_result_to_dict(r) for r in results],
        "last_scan": _last_scan_time,
    }

    for ws in _connected_websockets.copy():
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to WebSocket: {e}")
            if ws in _connected_websockets:
                _connected_websockets.remove(ws)


def run_dashboard(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the dashboard server in a background thread."""
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not installed, dashboard disabled")
        return

    app = create_app()

    def run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    logger.info(f"Dashboard started at http://{host}:{port}")


def get_dashboard_html() -> str:
    """Return the dashboard HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scalp Signal Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        h1 { font-size: 24px; }
        .status {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ff4444;
        }
        .status-dot.connected { background: #44ff44; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .stat-value { font-size: 28px; font-weight: bold; }
        .stat-label { font-size: 12px; color: #888; margin-top: 5px; }
        .signals-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }
        .signal-card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            border-left: 4px solid #888;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .signal-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .signal-card.bullish { border-left-color: #00ff88; }
        .signal-card.bearish { border-left-color: #ff4466; }
        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .symbol { font-size: 20px; font-weight: bold; }
        .signal-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .signal-badge.bullish { background: rgba(0,255,136,0.2); color: #00ff88; }
        .signal-badge.bearish { background: rgba(255,68,102,0.2); color: #ff4466; }
        .signal-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        .detail-item { font-size: 13px; }
        .detail-label { color: #888; }
        .detail-value { font-weight: 500; }
        .indicators {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        .indicator-chip {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            background: rgba(255,255,255,0.1);
        }
        .indicator-chip.bullish { background: rgba(0,255,136,0.15); color: #00ff88; }
        .indicator-chip.bearish { background: rgba(255,68,102,0.15); color: #ff4466; }
        .score-bar {
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }
        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #00ffff);
            border-radius: 2px;
            transition: width 0.3s;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #888;
        }
        .refresh-btn {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
        }
        .refresh-btn:hover { opacity: 0.9; }
        .refresh-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Scalp Signal Dashboard</h1>
            <div class="status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
                <button class="refresh-btn" id="refreshBtn" onclick="triggerScan()">Scan Now</button>
            </div>
        </header>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="totalSignals">0</div>
                <div class="stat-label">Total Signals</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="bullishCount">0</div>
                <div class="stat-label">Bullish</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="bearishCount">0</div>
                <div class="stat-label">Bearish</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="highConfidence">0</div>
                <div class="stat-label">High Confidence</div>
            </div>
        </div>

        <div class="signals-grid" id="signalsGrid">
            <div class="empty-state">
                <p>No signals yet. Click "Scan Now" to start scanning.</p>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let signals = [];

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('statusDot').classList.add('connected');
                document.getElementById('statusText').textContent = 'Connected';
            };

            ws.onclose = () => {
                document.getElementById('statusDot').classList.remove('connected');
                document.getElementById('statusText').textContent = 'Disconnected';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                if (event.data === 'ping') {
                    ws.send('pong');
                    return;
                }
                const data = JSON.parse(event.data);
                if (data.signals) {
                    signals = data.signals;
                    renderSignals();
                }
            };
        }

        function renderSignals() {
            const grid = document.getElementById('signalsGrid');

            document.getElementById('totalSignals').textContent = signals.length;
            document.getElementById('bullishCount').textContent = signals.filter(s => s.signal === 'BULLISH').length;
            document.getElementById('bearishCount').textContent = signals.filter(s => s.signal === 'BEARISH').length;
            document.getElementById('highConfidence').textContent = signals.filter(s => s.confidence === 'HIGH').length;

            if (signals.length === 0) {
                grid.innerHTML = '<div class="empty-state"><p>No signals yet. Click "Scan Now" to start scanning.</p></div>';
                return;
            }

            grid.innerHTML = signals.map(signal => `
                <div class="signal-card ${signal.signal.toLowerCase()}">
                    <div class="signal-header">
                        <span class="symbol">${signal.symbol}</span>
                        <span class="signal-badge ${signal.signal.toLowerCase()}">${signal.signal}</span>
                    </div>
                    <div class="signal-details">
                        <div class="detail-item">
                            <span class="detail-label">Price</span>
                            <div class="detail-value">$${signal.price < 1 ? signal.price.toFixed(6) : signal.price.toFixed(2)}</div>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Confidence</span>
                            <div class="detail-value">${signal.confidence}</div>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Score</span>
                            <div class="detail-value">${(signal.score * 100).toFixed(1)}%</div>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Confluence</span>
                            <div class="detail-value">${signal.confluence.bullish}B / ${signal.confluence.bearish}S</div>
                        </div>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${signal.score * 100}%"></div>
                    </div>
                    <div class="indicators">
                        ${Object.entries(signal.indicators).map(([name, ind]) => `
                            <span class="indicator-chip ${ind.signal.toLowerCase()}">${name}: ${typeof ind.value === 'number' ? ind.value.toFixed(2) : ind.value}</span>
                        `).join('')}
                    </div>
                </div>
            `).join('');
        }

        async function triggerScan() {
            const btn = document.getElementById('refreshBtn');
            btn.disabled = true;
            btn.textContent = 'Scanning...';

            try {
                const response = await fetch('/api/scan?timeframe=15m', { method: 'POST' });
                const data = await response.json();
                console.log('Scan complete:', data);
            } catch (e) {
                console.error('Scan error:', e);
            }

            btn.disabled = false;
            btn.textContent = 'Scan Now';
        }

        connect();
    </script>
</body>
</html>'''

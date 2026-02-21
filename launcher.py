#!/usr/bin/env python3
"""
Web-based launcher for Scalp Signal Bot.
Opens a browser page where you enter API keys and click Start.
"""
import os
import sys
import subprocess
import webbrowser
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs
import json

PORT = 8888
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")
BOT_PROCESS = None
BOT_OUTPUT = []  # Store recent output lines
MAX_OUTPUT_LINES = 100


def load_env():
    """Load existing .env file if it exists."""
    env = {"TELEGRAM_TOKEN": "", "DEEPSEEK_API_KEY": "", "BINANCE_API_KEY": "", "BINANCE_API_SECRET": ""}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env[key.strip()] = value.strip()
    return env


def save_env(telegram_token, deepseek_key, binance_key, binance_secret):
    """Save API keys to .env file."""
    with open(ENV_FILE, "w") as f:
        f.write(f"TELEGRAM_TOKEN={telegram_token}\n")
        f.write(f"DEEPSEEK_API_KEY={deepseek_key}\n")
        f.write(f"BINANCE_API_KEY={binance_key}\n")
        f.write(f"BINANCE_API_SECRET={binance_secret}\n")


def read_bot_output():
    """Read bot output in a background thread."""
    global BOT_PROCESS, BOT_OUTPUT
    while BOT_PROCESS and BOT_PROCESS.poll() is None:
        try:
            line = BOT_PROCESS.stdout.readline()
            if line:
                decoded = line.decode('utf-8', errors='replace').strip()
                if decoded:
                    BOT_OUTPUT.append(decoded)
                    # Keep only recent lines
                    if len(BOT_OUTPUT) > MAX_OUTPUT_LINES:
                        BOT_OUTPUT.pop(0)
                    print(f"[BOT] {decoded}")
        except Exception:
            break
    # Read any remaining output after process ends
    if BOT_PROCESS and BOT_PROCESS.stdout:
        try:
            remaining = BOT_PROCESS.stdout.read()
            if remaining:
                for line in remaining.decode('utf-8', errors='replace').split('\n'):
                    if line.strip():
                        BOT_OUTPUT.append(line.strip())
                        print(f"[BOT] {line.strip()}")
        except Exception:
            pass


def start_bot():
    """Start the bot process."""
    global BOT_PROCESS, BOT_OUTPUT
    BOT_OUTPUT = []  # Clear previous output
    env = os.environ.copy()

    # Load from .env
    saved = load_env()
    telegram_token = saved.get("TELEGRAM_TOKEN", "")
    deepseek_key = saved.get("DEEPSEEK_API_KEY", "")
    binance_key = saved.get("BINANCE_API_KEY", "")
    binance_secret = saved.get("BINANCE_API_SECRET", "")

    # Check for missing keys
    missing = []
    if not telegram_token:
        missing.append("Telegram Token")
    if not deepseek_key:
        missing.append("DeepSeek API Key")
    if not binance_key:
        missing.append("Binance API Key")
    if not binance_secret:
        missing.append("Binance API Secret")

    if missing:
        BOT_OUTPUT.append(f"ERROR: Missing: {', '.join(missing)}")
        BOT_OUTPUT.append("Please enter all API keys and click Save.")
        return None

    env["TELEGRAM_TOKEN"] = telegram_token
    env["DEEPSEEK_API_KEY"] = deepseek_key
    env["BINANCE_API_KEY"] = binance_key
    env["BINANCE_API_SECRET"] = binance_secret

    BOT_PROCESS = subprocess.Popen(
        [sys.executable, os.path.join(BASE_DIR, "bot.py")],
        cwd=BASE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Start output reader thread
    threading.Thread(target=read_bot_output, daemon=True).start()

    return BOT_PROCESS


def stop_bot():
    """Stop the bot process."""
    global BOT_PROCESS
    if BOT_PROCESS:
        BOT_PROCESS.terminate()
        BOT_PROCESS = None


class LauncherHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the launcher."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()

            env = load_env()
            is_running = BOT_PROCESS is not None and BOT_PROCESS.poll() is None

            status_class = "running" if is_running else "stopped"
            status_text = "Bot is RUNNING" if is_running else "Bot is STOPPED"
            button_action = "stop" if is_running else "start"
            button_class = "btn-stop" if is_running else "btn-start"
            button_text = "Stop Bot" if is_running else "Start Bot"

            # Get recent logs for display
            recent_logs = BOT_OUTPUT[-10:] if BOT_OUTPUT else []
            logs_html = "<br>".join(recent_logs) if recent_logs else "No logs yet. Start the bot to see output."

            # Check for errors in logs
            has_error = any("ERROR" in log or "error" in log.lower() or "Exception" in log for log in recent_logs)

            html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Scalp Signal Bot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .container {{
            background: #fff;
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 500px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            text-align: center;
            color: #1a1a2e;
            margin-bottom: 10px;
            font-size: 28px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .status {{
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 25px;
            font-weight: 600;
            font-size: 16px;
        }}
        .status.running {{
            background: #d4edda;
            color: #155724;
        }}
        .status.stopped {{
            background: #f8d7da;
            color: #721c24;
        }}
        label {{
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }}
        input {{
            width: 100%;
            padding: 14px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 16px;
            transition: border-color 0.3s;
        }}
        input:focus {{
            outline: none;
            border-color: #4CAF50;
        }}
        .help-text {{
            font-size: 12px;
            color: #888;
            margin-top: -15px;
            margin-bottom: 20px;
        }}
        .help-text a {{
            color: #4CAF50;
        }}
        button {{
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-bottom: 10px;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }}
        .btn-start {{
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
        }}
        .btn-stop {{
            background: linear-gradient(135deg, #f44336, #d32f2f);
            color: white;
        }}
        .btn-save {{
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
        }}
        .info {{
            margin-top: 25px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 10px;
            font-size: 13px;
            color: #666;
        }}
        .info strong {{
            color: #333;
        }}
        .icon {{
            font-size: 24px;
            vertical-align: middle;
            margin-right: 8px;
        }}
        .logs {{
            margin-top: 20px;
            padding: 15px;
            background: #1a1a2e;
            border-radius: 10px;
            font-size: 11px;
            color: #0f0;
            max-height: 200px;
            overflow-y: auto;
        }}
        .logs strong {{
            color: #fff;
        }}
        .logs pre {{
            margin: 10px 0 0 0;
            white-space: pre-wrap;
            word-break: break-all;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Scalp Signal Bot</h1>
        <p class="subtitle">Scans ALL Binance Futures pairs with AI</p>

        <div class="status {status_class}">
            {status_text}
        </div>

        <form method="POST" action="/save">
            <label>Telegram Bot Token</label>
            <input type="text" name="telegram_token" placeholder="Enter your Telegram bot token"
                   value="{env['TELEGRAM_TOKEN']}" required>
            <p class="help-text">Get from <a href="https://t.me/BotFather" target="_blank">@BotFather</a> on Telegram</p>

            <label>DeepSeek API Key</label>
            <input type="text" name="deepseek_key" placeholder="Enter your DeepSeek API key"
                   value="{env['DEEPSEEK_API_KEY']}" required>
            <p class="help-text">Get from <a href="https://platform.deepseek.com/" target="_blank">platform.deepseek.com</a></p>

            <label>Binance API Key</label>
            <input type="text" name="binance_key" placeholder="Enter your Binance API key"
                   value="{env['BINANCE_API_KEY']}" required>
            <p class="help-text">Get from <a href="https://www.binance.com/en/my/settings/api-management" target="_blank">Binance API Management</a></p>

            <label>Binance API Secret</label>
            <input type="password" name="binance_secret" placeholder="Enter your Binance API secret"
                   value="{env['BINANCE_API_SECRET']}" required>
            <p class="help-text">Keep this secret safe!</p>

            <button type="submit" class="btn-save">Save Keys</button>
        </form>

        <form method="POST" action="/{button_action}">
            <button type="submit" class="{button_class}">
                {button_text}
            </button>
        </form>

        <div class="logs" id="logs">
            <strong>Bot Output:</strong><br>
            <pre>{logs_html}</pre>
        </div>

        <div class="info">
            <strong>How to use:</strong><br>
            1. Enter your API keys above<br>
            2. Click "Save Keys"<br>
            3. Click "Start Bot"<br>
            4. Open Telegram and message your bot!
        </div>
    </div>
    <script>
        // Auto-refresh page every 5 seconds to show updated status
        setTimeout(function() {{ location.reload(); }}, 5000);
    </script>
</body>
</html>'''
            self.wfile.write(html.encode('utf-8'))

        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            is_running = BOT_PROCESS is not None and BOT_PROCESS.poll() is None
            self.wfile.write(json.dumps({"running": is_running}).encode())

        elif self.path == "/logs":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"logs": BOT_OUTPUT[-50:]}).encode())

        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode()
        params = parse_qs(post_data)

        if self.path == "/save":
            telegram_token = params.get("telegram_token", [""])[0]
            deepseek_key = params.get("deepseek_key", [""])[0]
            binance_key = params.get("binance_key", [""])[0]
            binance_secret = params.get("binance_secret", [""])[0]
            save_env(telegram_token, deepseek_key, binance_key, binance_secret)

            # Redirect back to main page
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        elif self.path == "/start":
            start_bot()
            time.sleep(1)  # Wait for bot to start
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        elif self.path == "/stop":
            stop_bot()
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        else:
            self.send_error(404)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    """Main entry point."""
    # Install dependencies if needed
    try:
        import aiohttp
        import telegram
    except ImportError:
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                       os.path.join(BASE_DIR, "requirements.txt")])

    # Create directories
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

    server = HTTPServer(("127.0.0.1", PORT), LauncherHandler)

    print("=" * 50)
    print("  SCALP SIGNAL BOT LAUNCHER")
    print("=" * 50)
    print(f"\n  Opening browser at: http://localhost:{PORT}")
    print("\n  Press Ctrl+C to close the launcher")
    print("=" * 50)

    # Open browser after a short delay
    def open_browser():
        time.sleep(1)
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        stop_bot()
        server.shutdown()


if __name__ == "__main__":
    main()

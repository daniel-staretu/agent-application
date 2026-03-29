"""
app.py — DataSage Flask Web Interface
======================================
Serves a chat UI at http://localhost:5000 backed by DataSageAgent.

Routes
------
  GET  /        — chat UI
  POST /chat    — {"message": "..."} → {"response": "...", "tools": [...]}
  POST /new     — reset short-term memory for this browser session
  GET  /memory  — long-term memory statistics
"""

import json
import os
import re
import secrets
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, session

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent import DataSageAgent

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# Server-side store: flask session ID → {"agent": ..., "lock": ...}
_AGENT_STORE: dict = {}
_store_lock = threading.Lock()

# Thread-local accumulator for tool calls made during a single chat() call
_tl = threading.local()


# ── WebDataSageAgent ──────────────────────────────────────────────────────────
# Subclass that captures tool calls without modifying agent.py

# Regex to extract [PLOT:/absolute/path/to/file.png] markers from tool output
_PLOT_MARKER_RE = re.compile(r'\[PLOT:([^\]]+)\]')

# Absolute path of the plots directory so we can convert to a URL-safe relative path
_PLOTS_DIR_ABS = str(Path("static/plots").resolve())


class WebDataSageAgent(DataSageAgent):
    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        if hasattr(_tl, "tool_calls"):
            _tl.tool_calls.append({
                "name": tool_name,
                "input_preview": json.dumps(tool_input, separators=(",", ":"))[:120],
            })

        result = super()._execute_tool(tool_name, tool_input)

        # Extract any [PLOT:path] markers, convert to web-accessible URLs,
        # and replace the markers with a human-readable note for Claude.
        plot_paths = _PLOT_MARKER_RE.findall(result)
        if plot_paths:
            for abs_path in plot_paths:
                filename = Path(abs_path).name
                url = f"/static/plots/{filename}"
                if hasattr(_tl, "plots"):
                    _tl.plots.append(url)
            # Replace markers with a note so Claude knows a plot was made
            result = _PLOT_MARKER_RE.sub(
                "(A plot was generated and will be displayed in the chat UI)",
                result,
            )

        return result


# ── Session helpers ───────────────────────────────────────────────────────────

def _get_session_data() -> dict:
    """Return (or lazily create) the server-side session data for this request."""
    if "sid" not in session:
        session["sid"] = secrets.token_hex(16)
    sid = session["sid"]
    with _store_lock:
        if sid not in _AGENT_STORE:
            _AGENT_STORE[sid] = {
                "agent": WebDataSageAgent(verbose=True),
                "lock": threading.Lock(),
            }
        return _AGENT_STORE[sid]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY is not set. Add it to your .env file."}), 500

    session_data = _get_session_data()

    with session_data["lock"]:
        _tl.tool_calls = []
        _tl.plots = []
        try:
            response_text = session_data["agent"].chat(user_message)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        tools_used = list(_tl.tool_calls)
        plots = list(_tl.plots)

    return jsonify({"response": response_text, "tools": tools_used, "plots": plots})


@app.route("/new", methods=["POST"])
def new_session():
    session_data = _get_session_data()
    with session_data["lock"]:
        session_data["agent"].new_session()
        new_sid = session_data["agent"].session_id
    return jsonify({"status": "ok", "session_id": new_sid})


@app.route("/memory")
def memory_stats():
    session_data = _get_session_data()
    return jsonify(session_data["agent"].memory_stats())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n[Error] ANTHROPIC_API_KEY is not set.")
        print("  Create a .env file with: ANTHROPIC_API_KEY=sk-ant-...\n")
    app.run(debug=True, threaded=True, port=5000)

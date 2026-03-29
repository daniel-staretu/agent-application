# DataSage — Intelligent Data Science Agent

DataSage is an agentic data science assistant built with the Anthropic Claude API. It uses a structured architecture with memory, planning, and tool use to complete multi-step data science tasks autonomously — accessible through a Flask web chat interface.

---

## Features

- **Conversational EDA** — load datasets, inspect shapes/dtypes/nulls, and get summary statistics
- **Python execution** — run arbitrary analysis code in a sandboxed environment
- **Visualization** — generate matplotlib/seaborn charts displayed inline in the chat
- **Persistent memory** — insights and results saved to SQLite; relevant context is retrieved at the start of every session
- **Multi-step planning** — for complex requests, DataSage produces an explicit plan before acting
- **Web UI** — clean chat interface with tool-call transparency (see which tools fired per response)

---

## Architecture

```
agent.py        — Main agent loop (profiling, planning, ReAct action loop)
app.py          — Flask web server + WebDataSageAgent subclass
memory.py       — Short-term (in-session) + long-term (SQLite) memory
tools.py        — Tool implementations: analyze_dataset, execute_python_code,
                  list_data_files, save_to_memory, search_memory
templates/      — Jinja2 HTML templates for the chat UI
static/         — CSS, JS, and generated plot images
data/           — Place datasets here (CSV, JSON, XLSX)
```

The agent follows a **THINK → TOOL CALL → OBSERVE → NEXT STEP** loop powered by Claude's tool use API. Long-term memory entries are retrieved by semantic relevance and injected into the system prompt at session start.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Or create a `.env` file:

```
ANTHROPIC_API_KEY=your_key_here
```

### 3. Run the web interface

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### 4. (Optional) Run the CLI agent directly

```bash
python main.py
```

---

## Example Tasks

- `"Load the Airbnb dataset and give me a full EDA report."`
- `"What are the top predictors of price in this dataset?"`
- `"Train a baseline logistic regression on the churn dataset and report accuracy and ROC-AUC."`
- `"Check this dataset for quality issues before I use it for modeling."`
- `"What did we find last time we analyzed this data?"` ← uses long-term memory

---

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- See `requirements.txt` for all Python dependencies

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude (Anthropic API) |
| Backend | Flask |
| Data | pandas, numpy, scipy |
| ML | scikit-learn |
| Visualization | matplotlib, seaborn |
| Memory | SQLite (via Python `sqlite3`) |
| Frontend | HTML/CSS/JS (Jinja2 templates) |

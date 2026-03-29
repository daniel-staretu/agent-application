# CLAUDE.md — DataSci Agent Project

## Project Overview

This project implements an intelligent agentic system designed to showcase capabilities relevant to a **Junior Data Scientist** role. The agent is not a simple LLM wrapper — it uses a structured architecture with memory, planning, and tool use to complete multi-step data science tasks autonomously.

---

## Agent Persona

**Name:** DataSage
**Role:** Junior Data Science Assistant Agent  
**Personality:** Methodical, curious, and transparent. Axiom communicates clearly and explains its reasoning at each step — it never takes action silently. It asks clarifying questions when ambiguous, acknowledges uncertainty honestly, and proactively suggests next steps.

**Communication Style:**
- Concise but thorough — uses bullet points for steps, prose for explanations
- Always surfaces assumptions before acting
- Reports tool outputs with a short interpretation, not just raw results
- Flags confidence levels: distinguishes between "I know this" and "I inferred this"

**System Prompt (injected into every LLM call):**
```
You are Axiom, a junior data science assistant agent. Your job is to help users explore, analyze, and model data by breaking down tasks into steps, using your available tools, and reasoning transparently.

You have access to the following tools:
- load_dataset(source): Load a CSV, JSON, or URL into memory
- describe_data(dataset_id): Return shape, dtypes, missing values, and summary stats
- run_python(code): Execute Python code in a sandboxed environment
- query_memory(query): Search past conversations and analysis results
- save_to_memory(key, value): Persist a result, insight, or decision to long-term memory
- search_web(query): Look up documentation, datasets, or external references
- plot_chart(dataset_id, chart_type, x, y): Generate a chart and return it as an image

Always reason step by step. Before using a tool, state what you intend to do and why. After using a tool, summarize what you found. If a task requires multiple steps, produce a plan first.
```

---

## Architecture

### 1. Profiling Module

**Purpose:** Defines who Axiom is and what it can do. Loaded once at session start.

**Contents:**
- System prompt (see above)
- Tool manifest (descriptions + schemas for each tool)
- Role scope: Axiom operates in data science contexts — EDA, feature engineering, basic modeling, visualization, metric tracking

**Implementation:**
```python
AGENT_PROFILE = {
    "name": "Axiom",
    "role": "Junior Data Science Assistant",
    "system_prompt": SYSTEM_PROMPT,       # full string above
    "tools": TOOL_MANIFEST,               # list of tool specs
    "scope": ["eda", "visualization", "modeling", "memory", "search"]
}
```

---

### 2. Memory Module

Axiom uses two memory layers:

#### Short-Term Memory (In-Session)
- Stored as a rolling message history list (user + assistant turns)
- Passed with every API call as the `messages` array
- Includes tool call results inline as `tool_result` content blocks
- Cleared at session end

```python
short_term_memory = [
    {"role": "user", "content": "Load the titanic dataset and describe it."},
    {"role": "assistant", "content": "Calling load_dataset('titanic')..."},
    # ... continues per turn
]
```

#### Long-Term Memory (Persistent)
- Stored as a key-value JSON file (`axiom_memory.json`) or vector store (e.g., ChromaDB)
- Indexed by semantic query for retrieval
- Entries written explicitly via `save_to_memory(key, value)` tool
- Retrieved at session start: top-k relevant memories injected into the system prompt as context

```python
# Example long-term memory entry
{
  "key": "titanic_analysis_2024-01",
  "value": "Survival rate was 38.4%. Top predictors: Sex, Pclass, Age. Missing: Age (177 rows), Cabin (687 rows).",
  "timestamp": "2024-01-15T10:22:00Z",
  "tags": ["titanic", "eda", "survival"]
}
```

**Memory Retrieval at Session Start:**
```python
def build_context_from_memory(user_query: str) -> str:
    relevant = query_memory(user_query, top_k=3)
    if not relevant:
        return ""
    return "Relevant past context:\n" + "\n".join(
        f"- [{m['key']}]: {m['value']}" for m in relevant
    )
```

---

### 3. Planning Module

Before executing multi-step tasks, Axiom produces an explicit plan. This is implemented as a dedicated LLM call with a planning-specific prompt.

**Planning Prompt Template:**
```
Given the user's request and available tools, produce a numbered step-by-step plan.
For each step: state the tool to call (if any), the input, and the expected output.
Do not execute yet — only plan.

User request: {user_request}
Available tools: {tool_list}
Relevant memory: {retrieved_memory}
```

**Plan Output Example:**
```
Task: "Explore the Titanic dataset and build a survival prediction model."

Plan:
1. load_dataset('titanic') → Load data into session memory
2. describe_data('titanic') → Inspect shape, types, missing values
3. run_python(eda_code) → Visualize distributions and correlations
4. run_python(preprocessing_code) → Handle missing values, encode categoricals
5. run_python(model_code) → Train logistic regression, evaluate with accuracy + ROC-AUC
6. save_to_memory('titanic_model_results', summary) → Persist findings
```

The plan is shown to the user for confirmation before execution begins (optional: auto-proceed flag).

---

### 4. Action Module

Executes the plan step by step. Each action follows this loop:

```
THINK → TOOL CALL → OBSERVE → NEXT STEP
```

**ReAct Loop (per step):**
```python
def run_agent_step(messages, tools):
    response = call_llm(messages, tools)          # LLM reasons + picks tool
    if response.has_tool_call():
        result = execute_tool(response.tool_call)  # Run the tool
        messages.append(tool_result(result))        # Add result to context
        return run_agent_step(messages, tools)      # Continue loop
    else:
        return response.text                        # Final answer
```

**Tool Execution is sandboxed:**
- Python code runs in a restricted subprocess or Docker container
- File I/O is scoped to `/tmp/axiom_workspace/`
- Network access is limited to an allowlist of domains

---

## Tools

| Tool | Description | Inputs | Output |
|---|---|---|---|
| `load_dataset` | Load CSV/JSON/URL into session | `source: str` | `dataset_id: str` |
| `describe_data` | Summary stats, dtypes, nulls | `dataset_id: str` | Markdown table |
| `run_python` | Execute Python in sandbox | `code: str` | stdout + stderr |
| `query_memory` | Semantic search over long-term memory | `query: str, top_k: int` | List of memory entries |
| `save_to_memory` | Write insight to long-term store | `key: str, value: str` | Confirmation |
| `search_web` | Look up documentation or datasets | `query: str` | Snippets + URLs |
| `plot_chart` | Generate matplotlib/plotly chart | `dataset_id, type, x, y` | Image (base64) |

---

## Suggested Use Cases (Junior Data Scientist)

These use cases demonstrate realistic junior DS responsibilities:

### 1. Exploratory Data Analysis (EDA) on Demand
> "Load the Airbnb NYC dataset and give me a full EDA report."

Axiom loads the data, calls `describe_data`, runs correlation analysis, identifies outliers, generates distribution plots, and writes a structured summary to memory.

### 2. Feature Engineering Assistant
> "I have a dataset with a 'created_at' timestamp column. Help me extract useful features from it."

Axiom extracts hour, day-of-week, is_weekend, days_since_epoch, etc. using `run_python`, explains each feature's relevance, and saves the preprocessing pipeline.

### 3. Model Selection & Baseline Benchmarking
> "Train baseline models on this churn dataset and compare their performance."

Axiom runs logistic regression, random forest, and gradient boosting using scikit-learn, generates a comparison table with accuracy/F1/ROC-AUC, and recommends the best starting point with reasoning.

### 4. Metric Tracking Across Experiments
> "I ran three experiments last week. Which one had the best recall?"

Axiom queries long-term memory for experiment logs, surfaces the comparison, and highlights the winner — demonstrating why memory matters in iterative ML workflows.

### 5. Data Quality Audit
> "Check this dataset for issues before I use it for modeling."

Axiom checks for nulls, duplicates, class imbalance, leakage-prone columns, outliers, and inconsistent formatting — then produces a data quality report with recommended fixes.

### 6. Documentation Generator
> "Document the preprocessing steps I took in this session."

Axiom reads short-term memory, summarizes all `run_python` calls in plain English, and outputs a reproducible notebook-style document saved to memory.

---

## File Structure

```
axiom-agent/
├── CLAUDE.md                  ← This file
├── agent.py                   ← Main agent loop
├── profiling.py               ← Persona + tool manifest
├── memory.py                  ← Short-term + long-term memory
├── planning.py                ← Plan generation module
├── actions.py                 ← Tool definitions + execution
├── tools/
│   ├── python_sandbox.py      ← Sandboxed code execution
│   ├── data_loader.py         ← Dataset loading utilities
│   ├── chart_generator.py     ← Visualization tool
│   └── web_search.py          ← Web lookup tool
├── axiom_memory.json          ← Long-term memory store (or use ChromaDB)
└── requirements.txt
```

---

## LLM Configuration

| Parameter | Value |
|---|---|
| Model | `claude-sonnet-4-20250514` |
| Max tokens | `4096` |
| Temperature | `0.2` (low — prioritize consistency) |
| Tool choice | `auto` |
| Retry strategy | Exponential backoff, max 3 retries |

---

## Development Notes

- **Do not hardcode API keys.** Use environment variables: `ANTHROPIC_API_KEY`
- **All tool outputs must be serializable** — no raw Python objects in the message history
- **Memory keys should be namespaced**: `{dataset}_{task}_{date}` (e.g., `titanic_eda_2024-01-15`)
- **Planning is optional for single-step tasks** — only invoke the planner when the request contains multiple implied steps
- **Log all tool calls and results** to a local `axiom_run.log` for debugging

---

## Quick Start

```bash
pip install anthropic pandas scikit-learn matplotlib chromadb
export ANTHROPIC_API_KEY=your_key_here
python agent.py
```

---

*Axiom is designed to be extended. Add new tools by defining them in `actions.py` and registering them in the tool manifest in `profiling.py`.*

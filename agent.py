"""
DataSage — Intelligent Data Science Agent
==========================================
Architecture
------------
  Profiling Module  : System prompt — defines DataSage's persona, capabilities,
                      and communication style.  Injected on every API call.

  Memory Module     : MemoryModule instance (memory.py).
                      Short-term  → current session message list.
                      Long-term   → SQLite for facts / insights across sessions.

  Planning Module   : Claude Opus 4.6 with adaptive thinking reasons about
                      the user's request, selects tools, and sequences actions.

  Action Module     : _execute_tool() dispatches each tool_use block returned
                      by Claude to the matching Python function in tools.py.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Optional

import anthropic

from memory import MemoryModule
from tools import (
    analyze_dataset,
    execute_python_code,
    list_data_files,
    save_to_memory,
    search_memory,
)


# ────────────────────────────────────────────────────────────────────────────── #
# Profiling Module — Persona and tool awareness                                  #
# ────────────────────────────────────────────────────────────────────────────── #

_BASE_SYSTEM_PROMPT = """\
You are DataSage, a senior data scientist and patient mentor designed to help \
junior data scientists grow their skills and solve real analytical problems.

## Personality & Communication Style
- Clear and educational: explain *why*, not just *what*
- Use analogies and real-world examples to build intuition
- Acknowledge uncertainty honestly; never fabricate statistics
- Celebrate small wins; break large problems into manageable steps
- Code explanations target a junior data scientist: annotate key lines

## Core Expertise
- Exploratory data analysis (EDA) and data quality assessment
- Feature engineering and preprocessing pipelines
- Statistical testing and inference
- Machine learning: model selection, evaluation, and interpretation
- Data visualisation best practices
- Python ecosystem: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn

## Working Principles
1. Always analyze data before recommending approaches — look at the actual shape,
   types, and distributions.
2. Prefer reproducible, well-commented code over one-liners.
3. Call `analyze_dataset` before writing modelling code — you need to see the data.
4. Use `execute_python_code` to verify your code actually runs; share the output.
5. Persist key facts (dataset descriptions, user preferences, model results) via
   `save_to_memory` so future sessions benefit from past context.
6. Search memory with `search_memory` before answering questions that may have
   prior context.

## Response Format
- Use Markdown headers and bullet lists for structure.
- Include code in fenced ``` python blocks.
- When presenting statistics, round to 4 significant figures.
- End complex answers with a "Next steps" section.
"""


# ────────────────────────────────────────────────────────────────────────────── #
# Tool Definitions (Planning Module input)                                        #
# ────────────────────────────────────────────────────────────────────────────── #

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "analyze_dataset",
        "description": (
            "Load a CSV or Excel file and return a comprehensive exploratory data analysis (EDA) "
            "report covering: shape, column types, missing values, numeric statistics (min, max, "
            "mean, std, skewness), high correlations, categorical cardinality, and duplicate rows. "
            "Always call this before writing modelling or cleaning code for a dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV or Excel file to analyse.",
                }
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "execute_python_code",
        "description": (
            "Execute Python code in an isolated subprocess and return stdout / stderr. "
            "pandas, numpy, scipy, and scikit-learn are pre-imported. "
            "Use this to: run data cleaning steps, compute statistics, train models, "
            "generate summary tables, or verify that a code snippet is correct. "
            "Always share the output with the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python code to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default 30).",
                    "default": 30,
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "list_data_files",
        "description": (
            "List all CSV and Excel files in a directory. "
            "Call this when the user asks what data files are available or "
            "when you need to discover datasets in a folder."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to search. Defaults to the current working directory.",
                    "default": ".",
                }
            },
            "required": [],
        },
    },
    {
        "name": "search_memory",
        "description": (
            "Search long-term memory for facts, insights, dataset descriptions, or user "
            "preferences stored in previous sessions. Call this when the user references "
            "prior work, asks about past analyses, or when context from a previous session "
            "may be relevant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing what to search for.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "save_to_memory",
        "description": (
            "Persist an important fact, insight, dataset description, model result, or "
            "user preference to long-term memory so it is available in future sessions. "
            "Use categories such as: 'dataset', 'model', 'preference', 'insight', 'result'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember.",
                },
                "category": {
                    "type": "string",
                    "description": "Category tag for organisation.",
                    "enum": ["dataset", "model", "preference", "insight", "result", "general"],
                    "default": "general",
                },
            },
            "required": ["content"],
        },
    },
]


# ────────────────────────────────────────────────────────────────────────────── #
# Agent                                                                           #
# ────────────────────────────────────────────────────────────────────────────── #

class DataSageAgent:
    """
    DataSage — Intelligent Data Science Agent.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Falls back to the ANTHROPIC_API_KEY environment variable.
    db_path : str
        Path for the SQLite long-term memory database.
    verbose : bool
        If True, print tool call names and results to stdout for transparency.
    """

    MODEL = "claude-opus-4-6"

    def __init__(
        self,
        api_key: Optional[str] = None,
        db_path: str = "datasage_memory.db",
        verbose: bool = True,
    ):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.memory = MemoryModule(db_path=db_path)
        self.session_id = str(uuid.uuid4())[:8]
        self.verbose = verbose

    # ── Profiling Module ──────────────────────────────────────────────────── #

    def _build_system_prompt(self, user_query: str = "") -> str:
        """Compose the system prompt with injected long-term memory context."""
        memory_context = self.memory.build_memory_context(user_query)
        if memory_context:
            return _BASE_SYSTEM_PROMPT + "\n\n" + memory_context
        return _BASE_SYSTEM_PROMPT

    # ── Action Module ─────────────────────────────────────────────────────── #

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch a tool call to the matching Python function."""
        if self.verbose:
            print(f"\n[Tool] {tool_name}({json.dumps(tool_input, indent=None)[:120]})")

        try:
            if tool_name == "analyze_dataset":
                result = analyze_dataset(tool_input["file_path"])

            elif tool_name == "execute_python_code":
                result = execute_python_code(
                    tool_input["code"],
                    timeout=tool_input.get("timeout", 30),
                )

            elif tool_name == "list_data_files":
                result = list_data_files(tool_input.get("directory", "."))

            elif tool_name == "search_memory":
                result = search_memory(self.memory, tool_input["query"])

            elif tool_name == "save_to_memory":
                result = save_to_memory(
                    self.memory,
                    tool_input["content"],
                    tool_input.get("category", "general"),
                )

            else:
                result = f"Unknown tool: {tool_name}"

        except Exception as exc:
            result = f"Tool execution error: {exc}"

        if self.verbose:
            preview = result[:200].replace("\n", " ")
            print(f"[Result] {preview}{'...' if len(result) > 200 else ''}\n")

        return result

    # ── Planning + Action loop ────────────────────────────────────────────── #

    def chat(self, user_message: str) -> str:
        """
        Send a message to DataSage and return the final text response.

        Uses the agentic tool-use loop:
          1. Send messages + tools to Claude (with adaptive thinking).
          2. If Claude returns tool_use blocks, execute them.
          3. Feed tool results back and call Claude again.
          4. Repeat until stop_reason == 'end_turn'.
        """
        # Add user message to short-term memory
        self.memory.add_message("user", user_message)

        # Build system prompt with relevant long-term context
        system = self._build_system_prompt(user_message)

        # Agentic loop (Planning → Action → repeat)
        messages = self.memory.get_messages()

        while True:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=8096,
                thinking={"type": "adaptive"},      # Planning Module: extended reasoning
                system=system,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            # Collect any text blocks for intermediate display
            text_blocks = [b for b in response.content if b.type == "text"]
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason == "end_turn" or not tool_use_blocks:
                # Final response — extract text
                final_text = "\n".join(b.text for b in text_blocks if b.text)

                # Store assistant reply in short-term memory
                self.memory.add_message("assistant", response.content)

                return final_text

            # ── Action Module: execute tools ──────────────────────────────
            # Append assistant turn (includes tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool and collect results
            tool_results = []
            for block in tool_use_blocks:
                result_text = self._execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            # Feed results back as a user message
            messages.append({"role": "user", "content": tool_results})

    # ── Session management ────────────────────────────────────────────────── #

    def new_session(self) -> None:
        """Start a fresh conversation (clears short-term memory)."""
        if self.memory.message_count() > 0:
            # Save a brief session record
            self.memory.save_session_summary(
                self.session_id,
                f"Session with {self.memory.message_count()} messages.",
            )
        self.memory.clear_session()
        self.session_id = str(uuid.uuid4())[:8]
        print(f"New session started (id={self.session_id})")

    def memory_stats(self) -> dict:
        return self.memory.all_stats()

"""
Action Module — Tool Implementations
=====================================
Each function here is a tool that the agent (Claude) can call.
They are registered in agent.py as JSON tool definitions.

Tools
-----
1. analyze_dataset      — EDA on a CSV / Excel file
2. execute_python_code  — Run arbitrary Python in a subprocess sandbox
3. list_data_files      — Discover CSV / Excel files in a directory
4. search_memory        — Keyword search over long-term memory (delegated to MemoryModule)
5. save_to_memory       — Persist a fact to long-term memory (delegated to MemoryModule)
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory import MemoryModule


# ────────────────────────────────────────────────────────────────────────────── #
# Tool 1 — analyze_dataset                                                       #
# ────────────────────────────────────────────────────────────────────────────── #

def analyze_dataset(file_path: str) -> str:
    """
    Load a CSV or Excel file and return a comprehensive EDA summary.

    Covers: shape, column types, missing values, numeric statistics,
    cardinality for categorical columns, and correlation highlights.
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        return "Error: pandas / numpy are not installed. Run `pip install pandas numpy`."

    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found — {file_path}"

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            return f"Error: unsupported file type '{suffix}'. Supported: .csv, .xlsx, .xls"
    except Exception as exc:
        return f"Error reading file: {exc}"

    lines: list[str] = []

    # ── Overview ──────────────────────────────────────────────────────────────
    lines += [
        f"# Dataset Analysis: {path.name}",
        "",
        "## Overview",
        f"- Rows: {df.shape[0]:,}",
        f"- Columns: {df.shape[1]}",
        f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB",
        "",
    ]

    # ── Column types ──────────────────────────────────────────────────────────
    lines.append("## Column Types")
    type_counts = df.dtypes.value_counts()
    for dtype, count in type_counts.items():
        lines.append(f"- {dtype}: {count} column(s)")
    lines.append("")

    # ── Missing values ────────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
    missing_df = missing_df[missing_df["missing"] > 0].sort_values("pct", ascending=False)

    if missing_df.empty:
        lines += ["## Missing Values", "No missing values found.", ""]
    else:
        lines += ["## Missing Values", f"Columns with missing data ({len(missing_df)} of {df.shape[1]}):"]
        for col, row in missing_df.iterrows():
            lines.append(f"  - {col}: {int(row['missing']):,} ({row['pct']}%)")
        lines.append("")

    # ── Numeric statistics ────────────────────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        lines.append("## Numeric Column Statistics")
        stats = numeric_df.describe().round(4)
        for col in stats.columns:
            col_stats = stats[col]
            skew = numeric_df[col].skew()
            lines.append(
                f"  {col}: min={col_stats['min']}, max={col_stats['max']}, "
                f"mean={col_stats['mean']:.4g}, std={col_stats['std']:.4g}, "
                f"skew={skew:.2f}"
            )
        lines.append("")

        # ── Correlation highlights ─────────────────────────────────────────
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            corr = corr_matrix.to_numpy().copy()
            np.fill_diagonal(corr, 0)
            corr_df = __import__('pandas').DataFrame(corr, index=corr_matrix.index, columns=corr_matrix.columns)
            high_corr = []
            for i, c1 in enumerate(corr_df.columns):
                for c2 in corr_df.columns[i + 1 :]:
                    val = corr_df.loc[c1, c2]
                    if val >= 0.7:
                        high_corr.append((c1, c2, round(float(val), 3)))
            if high_corr:
                lines.append("## High Correlations (|r| >= 0.70)")
                for c1, c2, val in sorted(high_corr, key=lambda x: -x[2]):
                    lines.append(f"  - {c1} ↔ {c2}: r={val}")
                lines.append("")

    # ── Categorical columns ───────────────────────────────────────────────────
    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        lines.append("## Categorical Columns")
        for col in cat_df.columns:
            n_unique = cat_df[col].nunique()
            top = cat_df[col].value_counts().head(3).to_dict()
            top_str = ", ".join(f"'{k}'({v})" for k, v in top.items())
            lines.append(f"  {col}: {n_unique} unique values  |  top: {top_str}")
        lines.append("")

    # ── Duplicate rows ────────────────────────────────────────────────────────
    n_dupes = df.duplicated().sum()
    lines += [
        "## Data Quality",
        f"- Duplicate rows: {n_dupes:,} ({n_dupes / len(df) * 100:.1f}%)",
        "",
    ]

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────── #
# Tool 2 — execute_python_code                                                   #
# ────────────────────────────────────────────────────────────────────────────── #

# Directory where generated plots are saved (served by Flask as /static/plots/)
PLOTS_DIR = Path("static") / "plots"

# Marker format embedded in stdout so app.py can extract plot paths
PLOT_MARKER = "[PLOT:{path}]"


def _build_preamble(plots_dir: str) -> str:
    """
    Build the code preamble injected before every user snippet.

    Configures matplotlib to:
      - Use the non-interactive Agg backend (no display needed)
      - Match the dark UI theme
      - Intercept plt.show() / plt.savefig() so figures are saved to
        plots_dir and their paths are printed as [PLOT:...] markers
    """
    return f"""
import warnings
warnings.filterwarnings('ignore')
import os as _os, uuid as _uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
try:
    from scipy import stats
except ImportError:
    pass
try:
    import sklearn
except ImportError:
    pass

# ── Dark theme to match the DataSage UI ───────────────────────────────────
_DARK = {{
    'figure.facecolor':  '#1e2235',
    'axes.facecolor':    '#1e2235',
    'axes.edgecolor':    '#2d3148',
    'axes.labelcolor':   '#94a3b8',
    'axes.titlecolor':   '#c7d2fe',
    'axes.grid':         True,
    'grid.color':        '#2d3148',
    'grid.linewidth':    0.6,
    'text.color':        '#e2e8f0',
    'xtick.color':       '#94a3b8',
    'ytick.color':       '#94a3b8',
    'legend.facecolor':  '#161926',
    'legend.edgecolor':  '#2d3148',
    'figure.figsize':    (9, 5),
    'figure.dpi':        110,
}}
plt.rcParams.update(_DARK)
sns.set_theme(style='dark', rc=_DARK)

# ── Auto-save helpers ──────────────────────────────────────────────────────
_PLOTS_DIR = r'{plots_dir}'
_os.makedirs(_PLOTS_DIR, exist_ok=True)

def _save_figure():
    fig = plt.gcf()
    axes = fig.get_axes()
    if not axes or not any(
        ax.has_data() or ax.lines or ax.collections or ax.patches or ax.images
        for ax in axes
    ):
        plt.close('all')
        return
    path = _os.path.join(_PLOTS_DIR, _uuid.uuid4().hex + '.png')
    _orig_savefig(path, dpi=130, bbox_inches='tight',
                  facecolor=plt.rcParams['figure.facecolor'])
    plt.close('all')
    print(f'[PLOT:{{path.replace(chr(92), "/")}}]')

# Intercept plt.show() — saves the figure instead of displaying it
plt.show = _save_figure

# Intercept plt.savefig() — redirects to plots_dir with a unique name
_orig_savefig = plt.savefig
def _patched_savefig(fname=None, *args, **kwargs):
    fname = _os.path.join(_PLOTS_DIR, _uuid.uuid4().hex + '.png')
    kwargs.setdefault('dpi', 130)
    kwargs.setdefault('bbox_inches', 'tight')
    kwargs.setdefault('facecolor', plt.rcParams['figure.facecolor'])
    _orig_savefig(fname, *args, **kwargs)
    plt.close('all')
    print(f'[PLOT:{{str(fname).replace(chr(92), "/")}}]')
plt.savefig = _patched_savefig
"""


def execute_python_code(code: str, timeout: int = 30) -> str:
    """
    Execute Python code in an isolated subprocess.

    matplotlib figures are automatically saved to static/plots/ and their
    paths are embedded in the returned string as [PLOT:...] markers so
    app.py can extract them and send them to the browser.

    NOTE: This does NOT provide strong sandboxing. It is designed for a
    trusted analyst environment, not a public-facing service.
    """
    plots_dir = str(PLOTS_DIR.resolve())
    preamble = _build_preamble(plots_dir)
    full_code = preamble + "\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(full_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        parts = []
        if stdout:
            parts.append(f"Output:\n{stdout[:3000]}")
        if stderr:
            parts.append(f"Errors / warnings:\n{stderr[:1000]}")
        if result.returncode != 0 and not stderr:
            parts.append(f"Process exited with code {result.returncode}")
        if not parts:
            parts.append("(Code ran successfully with no output)")

        return "\n\n".join(parts)

    except subprocess.TimeoutExpired:
        return f"Error: code execution timed out after {timeout} seconds."
    except Exception as exc:
        return f"Error launching subprocess: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ────────────────────────────────────────────────────────────────────────────── #
# Tool 3 — list_data_files                                                       #
# ────────────────────────────────────────────────────────────────────────────── #

def list_data_files(directory: str = "data") -> str:
    """
    List CSV and Excel files in *directory* (non-recursive).
    Returns a formatted table with file name, size, and last-modified time.
    """
    dirpath = Path(directory)
    if not dirpath.exists():
        return f"Error: directory not found — {directory}"
    if not dirpath.is_dir():
        return f"Error: '{directory}' is not a directory."

    patterns = ("*.csv", "*.xlsx", "*.xls")
    files = []
    for pat in patterns:
        files.extend(dirpath.glob(pat))

    if not files:
        return f"No CSV or Excel files found in '{directory}'."

    files.sort(key=lambda p: p.name.lower())
    lines = [f"Data files in '{dirpath.resolve()}':"]
    for f in files:
        stat = f.stat()
        size_kb = stat.st_size / 1024
        mtime = __import__("datetime").datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"  {f.name:<40} {size_kb:8.1f} KB   {mtime}")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────── #
# Tool 4 & 5 — memory tools (thin wrappers; MemoryModule injected at agent level)#
# ────────────────────────────────────────────────────────────────────────────── #

def search_memory(memory: "MemoryModule", query: str) -> str:
    """Search long-term memory for entries matching *query*."""
    results = memory.search(query, limit=6)
    if not results:
        return f"No memories found matching '{query}'."

    lines = [f"Memory results for '{query}':"]
    for r in results:
        ts = r["timestamp"][:16].replace("T", " ")
        lines.append(f"  [{r['category']} | {ts}] {r['content']}")
    return "\n".join(lines)


def save_to_memory(memory: "MemoryModule", content: str, category: str = "general") -> str:
    """Persist *content* to long-term memory under *category*."""
    row_id = memory.save(content, category)
    return f"Saved to memory (id={row_id}, category='{category}'): {content[:80]}{'...' if len(content) > 80 else ''}"

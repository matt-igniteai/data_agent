import asyncio
import logging
import os
import uuid
import ast
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field
from mcp.server import FastMCP  # mirrors your import for mcp_sql.py

# --- Logging Configuration (mirrors mcp_sql.py) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp_pandas")

# --- MCP Server ---
mcp = FastMCP(
    "pandas_mcp",
    "A server with tools to load CSV/Excel files and run safe pandas expressions."
)

# --- In-memory dataset registry (per-process session) ---
# Maps dataset_id -> {"df": DataFrame, "path": str, "type": str, "sheet": Optional[str]}
_DATASETS: Dict[str, Dict[str, Any]] = {}


# ---------- Helpers ----------
def _infer_file_type(path: str, explicit: Optional[str] = None) -> str:
    if explicit:
        et = explicit.lower()
        if et in {"csv", "xls", "xlsx"}:
            return et
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext in {".xls", ".xlsx"}:
        return "xlsx" if ext == ".xlsx" else "xls"
    # default to csv if unsure; callers should validate
    return "csv"


def _df_to_records(df: pd.DataFrame, limit: int = 100) -> List[Dict[str, Any]]:
    rows = df.to_dict(orient="records")
    if len(rows) > limit:
        return rows[:limit] + [{"notice": f"Result truncated to {limit} rows."}]
    return rows


# ---- Safe pandas expression validator/executor ----
# We allow a *single* Python expression that operates on the name `df`
# and uses a whitelist of DataFrame methods/attributes. No statements,
# no imports, no dunder access, no global builtins.

_ALLOWED_DF_ATTRS = {
    # selection / indexing
    "loc", "iloc", "at", "iat",
    # transforms
    "assign", "rename", "astype", "fillna", "dropna", "round",
    # sorting
    "sort_values", "sort_index",
    # summarization
    "groupby", "agg", "aggregate", "sum", "mean", "median", "min", "max",
    "count", "nunique", "value_counts", "describe",
    # reshaping
    "pivot_table", "reset_index", "set_index",
    # slicing helpers
    "head", "tail",
    # merging/joining (safe, in-memory)
    "merge", "join",
    # filtering
    "query",
}

_ALLOWED_OPERATORS = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.FloorDiv,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.USub, ast.UAdd, ast.Not,  # unary
)

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.Call,
    ast.Attribute, ast.Name, ast.Load, ast.Constant, ast.Subscript, ast.Slice,
    ast.Tuple, ast.List, ast.Dict, ast.keyword #, ast.Index  # `ast.Index` for Py<3.9 compatibility
)

def _validate_ast(node: ast.AST) -> None:
    # No statements at top level, only Expression
    if not isinstance(node, ast.Expression):
        raise ValueError("Only a single expression is allowed.")

    class _Guard(ast.NodeVisitor):
        def generic_visit(self, n: ast.AST) -> None:
            if isinstance(n, ast.AST) and not isinstance(n, _ALLOWED_NODES):
                raise ValueError(f"Disallowed syntax: {type(n).__name__}")
            super().generic_visit(n)

        def visit_Attribute(self, n: ast.Attribute) -> None:
            # Block dunder access
            if n.attr.startswith("__"):
                raise ValueError("Dunder attributes are not allowed.")
            # If base name is 'df', ensure attr is whitelisted or a column access will follow
            if isinstance(n.value, ast.Name) and n.value.id == "df":
                if n.attr not in _ALLOWED_DF_ATTRS:
                    # Allow `df.<col>` column-style access? Not valid for DataFrame.
                    # We require methods/attrs to be from whitelist.
                    raise ValueError(f"Attribute '{n.attr}' not allowed on df.")
            self.generic_visit(n)

        def visit_Name(self, n: ast.Name) -> None:
            if n.id not in {"df"}:
                raise ValueError("Only 'df' is available in expressions.")
            self.generic_visit(n)

        def visit_Call(self, n: ast.Call) -> None:
            # Disallow starargs/kwargs in Python <3.9 AST forms just in case
            for kw in n.keywords or []:
                if kw.arg and kw.arg.startswith("__"):
                    raise ValueError("Dunder keyword names are not allowed.")
            self.generic_visit(n)

        def visit_BinOp(self, n: ast.BinOp) -> None:
            if not isinstance(n.op, _ALLOWED_OPERATORS):
                raise ValueError("Operator not allowed.")
            self.generic_visit(n)

        def visit_BoolOp(self, n: ast.BoolOp) -> None:
            if not isinstance(n.op, _ALLOWED_OPERATORS):
                raise ValueError("Boolean operator not allowed.")
            self.generic_visit(n)

        def visit_Compare(self, n: ast.Compare) -> None:
            for op in n.ops:
                if not isinstance(op, _ALLOWED_OPERATORS):
                    raise ValueError("Comparison operator not allowed.")
            self.generic_visit(n)

        def visit_UnaryOp(self, n: ast.UnaryOp) -> None:
            if not isinstance(n.op, _ALLOWED_OPERATORS):
                raise ValueError("Unary operator not allowed.")
            self.generic_visit(n)

    _Guard().visit(node)


def _eval_pandas_expression(df: pd.DataFrame, expr: str) -> Union[pd.DataFrame, pd.Series, Any]:
    # Parse & validate
    try:
        parsed = ast.parse(expr, mode="eval")
        _validate_ast(parsed)
    except Exception as e:
        raise ValueError(f"Expression validation failed: {e}")

    # Execute with a barren global env and only 'df' in locals
    env_globals: Dict[str, Any] = {"__builtins__": {}}
    env_locals: Dict[str, Any] = {"df": df}

    try:
        compiled = compile(parsed, "<mcp_pandas>", "eval")
        result = eval(compiled, env_globals, env_locals)
        return result
    except Exception as e:
        raise RuntimeError(f"Expression execution failed: {e}")


# ---------- Tool Models ----------
class ChartRequest(BaseModel):
    chart_type: str = Field(..., description="One of 'bar', 'line', 'scatter', 'pie'.")
    data: List[Dict[str, Any]] = Field(..., description="Tabular rows to plot (records).")
    x_axis: Union[str, List[str]] = Field(..., description="Column(s) for X.")
    y_axis: Union[str, List[str]] = Field(..., description="Column(s) for Y (or values for pie).")
    title: Optional[str] = Field(None, description="Optional chart title.")
    color: Optional[str] = Field(None, description="Optional column for color grouping.")


default_file_path = os.getenv("CSV_FILE_PATH")

# ---------- MCP Tools ----------
@mcp.tool()
async def load_dataset(
    file_path: str = Field(default_file_path, description="Path to a CSV or Excel file (.csv, .xls, .xlsx)"),
    file_type: Optional[str] = Field(None, description="Optional explicit type: 'csv', 'xls', or 'xlsx'."),
    sheet_name: Optional[str] = Field(None, description="Excel sheet to load (if Excel)."),
    dataset_id: Optional[str] = Field(None, description="Optional ID to assign; otherwise auto-generated.")
) -> Dict[str, Any]:
    """
    Load a dataset into memory and register it under a session ID.

    Returns:
        {"dataset_id": str, "rows": int, "cols": int, "columns": [str], "notice": optional}
    """
    logger.info(f"Loading dataset from: {file_path} (type={file_type}, sheet={sheet_name})")

    def _blocking_load() -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ftype = _infer_file_type(file_path, file_type)
        if ftype == "csv":
            df = pd.read_csv(file_path)  # pandas CSV reader
        elif ftype in {"xls", "xlsx"}:
            # pandas chooses engine automatically (openpyxl/xlrd/calamine depending on availability)
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            raise ValueError("Unsupported file type. Use 'csv', 'xls', or 'xlsx'.")

        did = dataset_id or str(uuid.uuid4())
        _DATASETS[did] = {"df": df, "path": file_path, "type": ftype, "sheet": sheet_name}
        logger.info(f"Loaded dataset {did} with shape {df.shape}")
        return {
            "dataset_id": did,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(map(str, df.columns)),
        }

    try:
        return await asyncio.to_thread(_blocking_load)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def list_loaded_datasets() -> List[Dict[str, Any]]:
    """
    List dataset IDs currently loaded in this MCP process.
    """
    items = []
    for did, meta in _DATASETS.items():
        df = meta["df"]
        items.append({
            "dataset_id": did,
            "path": meta.get("path"),
            "type": meta.get("type"),
            "sheet": meta.get("sheet"),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
        })
    return items


@mcp.tool()
async def get_schema(
    dataset_id: str = Field(..., description="Dataset ID returned by load_dataset.")
) -> List[Dict[str, Any]]:
    """
    Return column schema: name + dtype.
    """
    logger.info(f"Fetching schema for dataset: {dataset_id}")
    try:
        meta = _DATASETS[dataset_id]
        df: pd.DataFrame = meta["df"]
        return [{"name": str(c), "dtype": str(dt)} for c, dt in df.dtypes.items()]
    except KeyError:
        return [{"error": f"Unknown dataset_id: {dataset_id}"}]
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def preview_dataset(
    dataset_id: str = Field(..., description="Dataset ID."),
    n: int = Field(5, description="Number of rows to preview.")
) -> Dict[str, Any]:
    """
    Return head/tail, shape, null counts—useful before asking questions.
    """
    logger.info(f"Preview dataset {dataset_id} (n={n})")
    try:
        df = _DATASETS[dataset_id]["df"]
        head_records = _df_to_records(df.head(n), limit=n)
        tail_records = _df_to_records(df.tail(n), limit=n)
        nulls = df.isna().sum().to_dict()
        return {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "head": head_records,
            "tail": tail_records,
            "null_counts": nulls,
        }
    except KeyError:
        return {"error": f"Unknown dataset_id: {dataset_id}"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def describe_dataset(
    dataset_id: str = Field(..., description="Dataset ID.")
) -> Dict[str, Any]:
    """
    Return basic descriptive statistics (numeric & object counts).
    """
    logger.info(f"Describe dataset {dataset_id}")
    try:
        df = _DATASETS[dataset_id]["df"]
        numeric_desc = df.describe(include=[float, int], datetime_is_numeric=True).to_dict()
        object_desc = df.describe(include=[object]).to_dict()
        return {
            "numeric": numeric_desc,
            "object": object_desc,
        }
    except KeyError:
        return {"error": f"Unknown dataset_id: {dataset_id}"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def run_pandas_expr(
    dataset_id: str = Field(..., description="Dataset ID."),
    expression: str = Field(
        ...,
        description=(
            "A single pandas expression using the name 'df', e.g.: "
            "df.groupby('country').agg({'sales':'sum'}).reset_index().sort_values('sales', ascending=False).head(20)"
        )
    ),
) -> Dict[str, Any]:
    """
    Execute a **restricted** pandas expression safely against the dataset.

    Security:
      - Only a single expression is allowed (no assignments/imports).
      - Only the name `df` is available.
      - DataFrame methods are restricted to a whitelist.
      - No double-underscore attributes.

    Returns:
        {
          "result_type": "dataframe|series|scalar",
          "columns": [...],             # when applicable
          "data": [ {..}, ... ],        # top 100 rows, with truncation notice if needed
          "repr": "stringified preview" # for Series/scalars
        }
    """
    logger.info(f"run_pandas_expr on {dataset_id}: {expression[:120]}...")
    try:
        df = _DATASETS[dataset_id]["df"]
    except KeyError:
        return {"error": f"Unknown dataset_id: {dataset_id}"}

    def _blocking_eval() -> Dict[str, Any]:
        try:
            out = _eval_pandas_expression(df, expression)
        except Exception as e:
            return {"error": str(e)}

        # Normalize outputs
        if isinstance(out, pd.DataFrame):
            return {
                "result_type": "dataframe",
                "columns": list(map(str, out.columns)),
                "data": _df_to_records(out),
            }
        elif isinstance(out, pd.Series):
            out_df = out.to_frame(name=str(out.name) if out.name is not None else "value").reset_index()
            return {
                "result_type": "series",
                "columns": list(map(str, out_df.columns)),
                "data": _df_to_records(out_df),
            }
        else:
            # Scalar or other object: return string repr plus minimal info
            return {
                "result_type": "scalar",
                "repr": repr(out),
            }

    return await asyncio.to_thread(_blocking_eval)


@mcp.tool()
async def generate_chart(request: ChartRequest) -> str:
    """
    Generate an interactive Plotly chart from tabular 'records' (e.g., the result of run_pandas_expr).

    Returns:
        HTML string (safe to embed) with include_plotlyjs='cdn'.
    """
    logger.info(f"generate_chart type={request.chart_type}")
    try:
        df = pd.DataFrame(request.data)

        if request.chart_type == "bar":
            fig = px.bar(df, x=request.x_axis, y=request.y_axis, title=request.title, color=request.color)
        elif request.chart_type == "line":
            fig = px.line(df, x=request.x_axis, y=request.y_axis, title=request.title, color=request.color)
        elif request.chart_type == "scatter":
            fig = px.scatter(df, x=request.x_axis, y=request.y_axis, title=request.title, color=request.color)
        elif request.chart_type == "pie":
            # Plotly pie uses names/values semantics
            fig = px.pie(df, names=request.x_axis, values=request.y_axis, title=request.title)
        else:
            return '{"error":"Unsupported chart type"}'

        # compact embed (loads plotly.js from CDN)
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as e:
        return f'{{"error": "Failed to generate chart.", "details": "{str(e)}"}}'


# --- Main Execution Block (mirrors mcp_sql.py) ---
if __name__ == "__main__":
    logger.info("Starting Pandas MCP Server...")
    mcp.run(transport="stdio")

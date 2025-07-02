import asyncio
import logging
import re
from dotenv import load_dotenv
import os
import pyodbc
from mcp.server import FastMCP
from pydantic import Field, BaseModel
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any, Optional, Union

# Load environment variables from .env file
load_dotenv()


# --- Logging Configuration ---
# Set up a structured logger to monitor server operations.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_sql")

# --- Database Configuration ---

def get_db_config() -> Dict[str, Optional[str]]:
    """
    Retrieves database configuration from environment variables.
    This function ensures that all necessary components for the connection
    string are available.
    """
    config = {
        "Driver": os.getenv("MSSQL_DRIVER", "{ODBC Driver 17 for SQL Server}"),
        "server": os.getenv("MSSQL_SERVER"),
        "user": os.getenv("MSSQL_USER"),
        "password": os.getenv("MSSQL_PASSWORD"),
        "database": os.getenv("MSSQL_DATABASE")
    }
    # Check for essential configuration parameters
    if not all([config["server"], config["user"], config["password"], config["database"]]):
        logger.error("Missing required database configuration. Please check .env file.")
        logger.error("MSSQL_SERVER, MSSQL_USER, MSSQL_PASSWORD, and MSSQL_DATABASE are required.")
        raise ValueError("Missing required database configuration.")
    
    return config

def get_connection_string() -> str:
    """
    Constructs the pyodbc connection string from the configuration.
    Using a dedicated function for this keeps the connection logic centralized.
    """
    config = get_db_config()
    # TrustServerCertificate=yes is often needed for development environments
    # or when connecting to SQL Server instances without a trusted certificate.
    return (
        f"DRIVER={config['Driver']};"
        f"SERVER={config['server']};"
        f"DATABASE={config['database']};"
        f"UID={config['user']};"
        f"PWD={config['password']};"
        "TrustServerCertificate=yes;"
    )


# --- MCP Server Initialization ---
# The FastMCP class provides the foundation for our tool server.
mcp = FastMCP(
    "sql_server_mcp",
    "A server with tools to interact with a Microsoft SQL Server database."
)


# --- MCP Tools ---

@mcp.tool()
async def connect_to_database() -> str:
    """
    Tests the connection to the SQL Server database using the provided credentials.
    
    This tool is useful for verifying that the server configuration and database
    credentials are correct without performing any actual queries.

    Returns:
        A string indicating if the connection was successful or not.
    """
    logger.info("Attempting to connect to the database...")
    
    def _blocking_connect():
        """This function contains the blocking I/O call."""
        try:
            conn_str = get_connection_string()
            # The 'with' statement ensures the connection is closed automatically.
            with pyodbc.connect(conn_str, timeout=5) as conn:
                logger.info("Database connection test successful.")
                return "Successfully connected to the database."
        except pyodbc.Error as ex:
            logger.error(f"Database connection failed: {ex}")
            # Propagate the error to be handled by the async wrapper
            raise

    try:
        # Run the blocking database operation in a separate thread
        result = await asyncio.to_thread(_blocking_connect)
        return result
    except Exception as e:
        return f"Connection Error: {e}"


@mcp.tool()
async def list_tables() -> List[str]:
    """
    Retrieves a list of all user-defined table names from the database.
    
    It queries the database's information schema to find tables that are
    of the 'BASE TABLE' type, excluding system views and other objects.

    Returns:
        A list of strings, where each string is a table name.
    """
    logger.info("Executing 'list_tables' tool.")
    query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME;"

    def _blocking_fetch():
        try:
            conn_str = get_connection_string()
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                tables = [row.TABLE_NAME for row in cursor.fetchall()]
                logger.info(f"Found {len(tables)} tables.")
                return tables
        except pyodbc.Error as ex:
            logger.error(f"Error listing tables: {ex}")
            raise
    
    try:
        return await asyncio.to_thread(_blocking_fetch)
    except Exception as e:
        return [f"Error: {e}"]


@mcp.tool()
async def get_table_schema(table_name: str = Field(..., description="The name of the table to inspect.")) -> List[Dict[str, Any]]:
    """
    Retrieves the schema information for a specific table.

    Provides details for each column, including its name, data type, and
    maximum length (if applicable). This is useful for understanding
    table structures before querying them.

    Args:
        table_name: The name of the table you want the schema for.

    Returns:
        A list of dictionaries, where each dictionary represents a column
        and contains its 'name', 'type', and 'max_length'.
    """
    logger.info(f"Executing 'get_table_schema' for table: {table_name}")
    query = """
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION;
    """
    
    def _blocking_fetch_schema():
        try:
            conn_str = get_connection_string()
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(query, table_name)
                columns = [
                    {"name": row.COLUMN_NAME, "type": row.DATA_TYPE, "max_length": row.CHARACTER_MAXIMUM_LENGTH}
                    for row in cursor.fetchall()
                ]
                logger.info(f"Retrieved schema for {len(columns)} columns from table '{table_name}'.")
                return columns
        except pyodbc.Error as ex:
            logger.error(f"Error getting schema for table '{table_name}': {ex}")
            raise

    try:
        schema = await asyncio.to_thread(_blocking_fetch_schema)
        if not schema:
            return [{"error": f"Table '{table_name}' not found or has no columns."}]
        return schema
    except Exception as e:
        return [{"error": f"An unexpected error occurred: {e}"}]


# --- SQL Query Validation ---
FORBIDDEN_SQL_KEYWORDS = [
    "INSERT", "DELETE", "UPDATE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "EXEC", "MERGE", "GRANT", "REVOKE"
]

def sql_query_validation(sql_query: str) -> bool:
    """
    Validates that the provided SQL query is a safe SELECT statement.
    Ensures the query does not contain forbidden SQL keywords that
    could modify the database or execute other harmful operations.

    Returns:
        True if the query is safe for execution; False otherwise.
    """
    normalized = sql_query.strip().upper()
    if not normalized.startswith("SELECT"):
        return False
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalized):
            return False
    return True


@mcp.tool()
async def execute_sql_query(
    sql_query: str = Field(..., description="The SQL query to execute against the database.")
    ) -> List[Dict[str, Any]]:
    """
    Executes a read-only (SELECT) SQL query and returns the results.
    
    This tool allows for fetching data from any table. To protect against
    very large results, it automatically limits the output to 100 rows.

    Args:
        sql_query: The complete SQL SELECT statement to run.

    Returns:
        A list of dictionaries, where each dictionary is a row from the result set.
        Returns an error message in case of a syntax or execution error.
    """
    logger.info(f"Executing user-provided SQL query: {sql_query[:100]}...")
    
    # Strong validation to ensure it's a safe read-only SELECT query
    if not sql_query_validation(sql_query):
        logger.warning(f"Blocked unsafe or non-SELECT query attempt: {sql_query}")
        return [{"error": "This tool only supports simple read-only SELECT queries without harmful operations."}]

    def _blocking_run_query():
        try:
            conn_str = get_connection_string()
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                
                if cursor.description:
                    columns = [column[0] for column in cursor.description]
                    # Fetch up to 101 rows to see if there are more than the limit
                    rows = cursor.fetchmany(101) 
                    results = [dict(zip(columns, row)) for row in rows]
                    
                    if len(results) > 100:
                        logger.warning("Query result exceeded 100 rows, truncating.")
                        return results[:100] + [{"notice": "Result truncated to 100 rows."}]

                    logger.info(f"Query executed successfully, returning {len(results)} rows.")
                    return results
                else:
                    # This case handles queries that don't return rows (e.g., within a stored procedure)
                    logger.info("Query executed but returned no rows.")
                    return [{"status": "Query executed, no rows returned."}]

        except pyodbc.Error as ex:
            logger.error(f"Error executing SQL query: {ex}")
            raise # Re-raise to be caught by the outer try/except block

    try:
        return await asyncio.to_thread(_blocking_run_query)
    except Exception as e:
        # Return a dictionary with an error key for better machine readability
        return [{"error": f"Failed to execute query. SQL Server says: {e}"}]



# # ---- Updated Chart Data Model ----
# class ChartRequest(BaseModel):
#     """
#     Defines the data structure for a request to generate a chart.
#     This model provides a structured way to specify the chart type,
#     the data to be plotted, and various customization options.
#     """
#     chart_type: str = Field(
#         ...,
#         description="The type of chart to generate. Must be one of 'bar', 'pie', 'line', or 'scatter'."
#     )
#     data: List[Dict[str, Any]] = Field(
#         ...,
#         description="The dataset to be plotted, provided as a list of dictionaries, where each dictionary represents a row of data."
#     )
#     x_axis: Union[str, List[str]] = Field(
#         ...,
#         description="The key(s) from the data dictionaries to be used for the x-axis. For most charts, this will be a single string. For some chart types, it could be a list of strings."
#     )
#     y_axis: Union[str, List[str]] = Field(
#         ...,
#         description="The key(s) from the data dictionaries to be used for the y-axis. For most charts, this will be a single string. For some chart types, it could be a list of strings."
#     )
#     title: Optional[str] = Field(
#         None,
#         description="An optional title for the chart. If not provided, the chart will be generated without a title."
#     )
#     color: Optional[str] = Field(
#         None,
#         description="An optional key from the data dictionaries to be used for color-coding the chart's data points. This can be used to differentiate data in bar, line, and scatter plots."
#     )


# @mcp.tool()
# async def generate_chart(request: ChartRequest) -> str:
#     """
#     Generates an interactive chart based on the provided data and specifications.

#     This tool is designed to take a structured request, including the chart type,
#     the data to be plotted, and aesthetic parameters like title and color,
#     and returns an HTML string containing an interactive Plotly chart.

#     As an LLM, you should use this tool whenever a user asks to visualize data.
#     First, gather all the necessary information from the user:
#     1. The type of chart they want (e.g., 'bar', 'pie', 'line', 'scatter').
#     2. The data they want to visualize.
#     3. The columns or categories to be used for the x-axis and y-axis.
#     4. Any optional preferences like a chart title or a column to base colors on.

#     Once you have this information, construct the 'ChartRequest' object and call this tool.

#     For example, if a user says: "Can you make a bar chart showing the population of London, Paris, and Tokyo?",
#     you should formulate a request like this:
#     request = ChartRequest(
#         chart_type='bar',
#         data=[
#             {'city': 'London', 'population': 8900000},
#             {'city': 'Paris', 'population': 2140000},
#             {'city': 'Tokyo', 'population': 13960000}
#         ],
#         x_axis='city',
#         y_axis='population',
#         title='Population of Major Cities'
#     )
#     generate_chart(request)

#     Args:
#         request: A ChartRequest object containing all the necessary information
#                  to generate the chart.

#     Returns:
#         A string containing the HTML and JavaScript for an interactive Plotly chart,
#         or a JSON string with an error message if the chart type is unsupported.
#     """
#     try:
#         df = pd.DataFrame(request.data)

#         fig = None
#         if request.chart_type == "bar":
#             fig = px.bar(df, x=request.x_axis, y=request.y_axis, title=request.title, color=request.color)
#         elif request.chart_type == "pie":
#             # Note: Plotly's pie chart uses 'names' and 'values' instead of 'x' and 'y'.
#             fig = px.pie(df, names=request.x_axis, values=request.y_axis, title=request.title)
#         elif request.chart_type == "line":
#             fig = px.line(df, x=request.x_axis, y=request.y_axis, title=request.title, color=request.color)
#         elif request.chart_type == "scatter":
#             fig = px.scatter(df, x=request.x_axis, y=request.y_axis, title=request.title, color=request.color)
#         # To add more chart types, simply add another 'elif' block here.

#         if fig:
#             return fig.to_html(full_html=False, include_plotlyjs='cdn')
#         else:
#             return '{"error": "Unsupported chart type"}'
#     except Exception as e:
#         return f'{{"error": "Failed to generate chart.", "details": "{str(e)}"}}'
    


# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Starting SQL Server MCP Server...")
    # The `mcp.run()` function starts the server and listens for requests.
    # 'stdio' transport is used for communication over standard input/output.
    mcp.run(transport='stdio')



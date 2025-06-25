import asyncio
import logging
from dotenv import load_dotenv
import os
import pyodbc
from mcp.server import FastMCP
from pydantic import Field
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()


# --- Logging Configuration ---
# Set up a structured logger to monitor server operations.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_server")

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
    "ignite_sql_server_mcp",
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
    
    # Basic validation to ensure it's a read-only query
    if not sql_query.strip().upper().startswith("SELECT"):
        logger.warning(f"Blocked non-SELECT query attempt: {sql_query}")
        return [{"error": "This tool only supports SELECT queries."}]

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


# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Starting SQL Server MCP Server...")
    # The `mcp.run()` function starts the server and listens for requests.
    # 'stdio' transport is used for communication over standard input/output.
    mcp.run(transport='stdio')



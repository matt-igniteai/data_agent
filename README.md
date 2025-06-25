# NL2SQL MCP Project

A Natural Language to SQL (NL2SQL) project that uses Model Context Protocol (MCP) to enable natural language queries against Microsoft SQL Server databases through an AI chatbot interface powered by Anthropic's Claude.

## Project Overview

This project consists of two main components:

1. **MCP Server** (`mcp_server.py`) - Provides database connectivity and SQL execution tools
2. **MCP Chatbot Client** (`mcp_chatbot.py`) - Interactive chatbot that processes natural language queries

The system allows users to ask questions in natural language about their database, which are then converted to SQL queries and executed against a Microsoft SQL Server database.

## Features

- **Database Connection Testing** - Verify connectivity to SQL Server
- **Schema Exploration** - List tables and inspect table schemas
- **Natural Language Queries** - Convert plain English to SQL using Claude AI
- **Safe Query Execution** - Read-only SELECT queries with result limiting
- **Multi-Server Support** - Connect to multiple MCP servers simultaneously

## Prerequisites

- Python 3.12 or higher
- Microsoft SQL Server database
- UV package manager
- Valid Anthropic API key

## Installation

1. Clone the repository:
```bash
git clone https://igniteaipartners@dev.azure.com/igniteaipartners/Generic%20Research%20and%20Development/_git/nl2sql_mcp
cd nl2sql_mcp
```

2. Install dependencies using UV:
```bash
uv sync
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with your database credentials:

```env
# SQL Server Configuration
MSSQL_SERVER=your-server-name
MSSQL_DATABASE=your-database-name
MSSQL_USER=your-username
MSSQL_PASSWORD=your-password
MSSQL_DRIVER={ODBC Driver 17 for SQL Server}

# Anthropic API Key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

### Server Configuration

The `server_config.json` file defines MCP server connections. The default configuration includes:

- `ignite_sql_server_mcp` - The SQL Server MCP server
- `filesystem` - File system operations
- `fetch` - Web content fetching

## Usage

### Running the MCP Chatbot

To start the interactive chatbot with UV package management:

```bash
uv run mcp_chatbot.py
```

This will:
1. Connect to all configured MCP servers
2. Initialize the Claude AI client
3. Start an interactive chat loop

### Example Queries

Once the chatbot is running, you can ask questions like:

- "What tables are available in the database?"
- "Show me the schema for the customers table"
- "How many orders were placed last month?"
- "List the top 10 customers by revenue"

Type `quit` to exit the chatbot.

### Running Individual Components

**MCP Server only:**
```bash
uv run mcp_server.py
```

**Testing database connection:**
The server provides tools to test connectivity and explore your database schema before running queries.

## MCP Inspector Integration

To connect with the Model Context Protocol Inspector for debugging and development:

1. Install the MCP Inspector:
```bash
npm install -g @modelcontextprotocol/inspector
```

2. Run the inspector with your MCP server:
```bash
mcp-inspector uv run mcp_server.py
```

3. Open your browser to the provided URL to interact with the MCP server through the web interface.

The inspector allows you to:
- Test individual MCP tools
- View tool schemas and descriptions
- Debug server responses
- Monitor server logs

## Available MCP Tools

The SQL Server MCP provides these tools:

- **connect_to_database** - Test database connectivity
- **list_tables** - Get all table names
- **get_table_schema** - Inspect table structure
- **execute_sql_query** - Run SELECT queries (read-only)

## Security Features

- **Read-only queries** - Only SELECT statements are allowed
- **Result limiting** - Query results are limited to 100 rows
- **SQL injection protection** - Basic validation of query types
- **Connection timeout** - Database connections have timeout limits

## Troubleshooting

### Common Issues

1. **Database connection failed:**
   - Verify your `.env` file configuration
   - Ensure SQL Server is accessible
   - Check firewall settings

2. **Missing dependencies:**
   - Run `uv sync` to install all dependencies
   - Ensure Python 3.12+ is installed

3. **MCP server connection issues:**
   - Check `server_config.json` configuration
   - Verify UV is properly installed

### Logging

The application uses structured logging. Check console output for detailed information about:
- Database connections
- Query executions
- MCP server communications
- Error messages

## Development

The project uses UV for dependency management. Key files:

- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Locked dependency versions
- `server_config.json` - MCP server configurations

## Contributing

When contributing to this project:

1. Ensure all database credentials are properly configured in `.env`
2. Test both MCP server and client functionality
3. Follow existing code patterns and logging practices
4. Update documentation for any new features

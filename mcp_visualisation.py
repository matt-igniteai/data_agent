import logging
from dotenv import load_dotenv
from mcp.server import FastMCP
from pydantic import Field, BaseModel
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import plotly.express as px

# Load environment variables from .env file


load_dotenv()

# --- Logging Configuration ---
# Set up a structured logger to monitor server operations.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_visualisation")

# --- MCP Server Initialization ---
# The FastMCP class provides the foundation for our tool server.
mcp = FastMCP(
    "visualisation_mcp_server",
    "A server with tools to create interactive data visualisations."
)


# ---- Updated Chart Data Model ----
class ChartRequest(BaseModel):
    """
    Defines the data structure for a request to generate a chart.
    This model provides a structured way to specify the chart type,
    the data to be plotted, and various customization options.
    """
    chart_type: str = Field(
        ...,
        description="The type of chart to generate. Must be one of 'bar', 'pie', 'line', or 'scatter'."
    )
    data: List[Dict[str, Any]] = Field(
        ...,
        description="The dataset to be plotted, provided as a list of dictionaries, where each dictionary represents a row of data."
    )
    x_axis: Union[str, List[str]] = Field(
        ...,
        description="The key(s) from the data dictionaries to be used for the x-axis. For most charts, this will be a single string. For some chart types, it could be a list of strings."
    )
    y_axis: Union[str, List[str]] = Field(
        ...,
        description="The key(s) from the data dictionaries to be used for the y-axis. For most charts, this will be a single string. For some chart types, it could be a list of strings."
    )
    title: Optional[str] = Field(
        None,
        description="An optional title for the chart. If not provided, the chart will be generated without a title."
    )
    color: Optional[str] = Field(
        None,
        description="An optional key from the data dictionaries to be used for color-coding the chart's data points. This can be used to differentiate data in bar, line, and scatter plots."
    )


@mcp.tool()
async def create_chart(request: ChartRequest) -> str:
    """
    Generates an interactive chart based on the provided data and specifications.

    This tool is designed to take a structured request, including the chart type,
    the data to be plotted, and aesthetic parameters like title and color,
    and returns an HTML string containing an interactive Plotly chart.

    As an LLM, you should use this tool whenever a user asks to visualize data.
    First, gather all the necessary information from the user:
    1. The type of chart they want (e.g., 'bar', 'pie', 'line', 'scatter').
    2. The data they want to visualize.
    3. The columns or categories to be used for the x-axis and y-axis.
    4. Any optional preferences like a chart title or a column to base colors on.

    Once you have this information, construct the 'ChartRequest' object and call this tool.

    For example, if a user says: "Can you make a bar chart showing the population of London, Paris, and Tokyo?",
    you should formulate a request like this:
    request = ChartRequest(
        chart_type='bar',
        data=[
            {'city': 'London', 'population': 8900000},
            {'city': 'Paris', 'population': 2140000},
            {'city': 'Tokyo', 'population': 13960000}
        ],
        x_axis='city',
        y_axis='population',
        title='Population of Major Cities'
    )
    generate_chart(request)

    Args:
        request: A ChartRequest object containing all the necessary information
                 to generate the chart.

    Returns:
        A string containing the HTML and JavaScript for an interactive Plotly chart,
        or a JSON string with an error message if the chart type is unsupported.
    """
    try:
        df = pd.DataFrame(request.data)

        fig = None
        if request.chart_type == "bar":
            fig = px.bar(df, x=request.x_axis, y=request.y_axis, title=request.title, color=request.color)
        elif request.chart_type == "pie":
            # Note: Plotly's pie chart uses 'names' and 'values' instead of 'x' and 'y'.
            fig = px.pie(df, names=request.x_axis, values=request.y_axis, title=request.title)
        elif request.chart_type == "line":
            fig = px.line(df, x=request.x_axis, y=request.y_axis, title=request.title, color=request.color)
        elif request.chart_type == "scatter":
            fig = px.scatter(df, x=request.x_axis, y=request.y_axis, title=request.title, color=request.color)
        # To add more chart types, simply add another 'elif' block here.

        if fig:
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        else:
            return '{"error": "Unsupported chart type"}'
    except Exception as e:
        return f'{{"error": "Failed to generate chart.", "details": "{str(e)}"}}'
    
# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Starting Data Visualisation MCP Server...")
    # The `mcp.run()` function starts the server and listens for requests.
    # 'stdio' transport is used for communication over standard input/output.
    mcp.run(transport='stdio')
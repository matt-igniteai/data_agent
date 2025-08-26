
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import json
import asyncio
import logging

# --- Configuration & Initialization --- 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_chatbot")

# Load environment variables from a .env file
load_dotenv()

class ToolDefinition(TypedDict):
    "To extract definition of tools from server"
    name: str
    description: str
    input_schema: dict


# --- Main Chatbot Class ---

class MCP_ChatBot:
    """
    A chatbot that connects to multiple MCP servers and uses their tools
    to answer user queries with the help of the Anthropic Claude 3 model.
    """

    def __init__(self):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = [] # new - all the sessions are being connected to
        self.exit_stack = AsyncExitStack() # new - we have this for reading and writing as well as managing the entire connection to the session
        self.anthropic = Anthropic()
        self.available_tools: List[ToolDefinition] = [] # new - All the available tools 
        self.tool_to_session: Dict[str, ClientSession] = {} # new - all the tools from particular sessions

        """This class is not production ready but the focus here is to make sure that we correctly map a tool."""

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        logger.info(f"\nAttempting to connect to server: {server_name}\n") # ----- NEW
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            ) # new - to manage the entire connection to the session
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            ) # new - for reading and writing a connection to the session

            # here we initialise the session
            await session.initialize()
            self.sessions.append(session)
            
            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            logger.info(f"\nConnected: {server_name} mcp server with tools: {[t.name for t in tools]}\n") # ----- NEW

            # append tolls from servers that we wanted to connect to
            for tool in tools: # new
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self): # new
        """Connect to all configured MCP servers."""
        try:
            # go ahead and read from server config file, for server connection
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            if not servers:
                logger.warning("No MCP servers found in server_config.json.")
                return

            # connect all servers
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)

        except Exception as e:
            logger.info(f"Error loading server configuration: {e}")
            raise
    
    async def process_query(self, query):
        messages = [{'role':'user', 'content':query}] # query comes from user
        response = self.anthropic.messages.create(max_tokens = 2024,
                                      model = 'claude-3-7-sonnet-20250219', 
                                      tools = self.available_tools,
                                      messages = messages)
        
        process_query = True
        while process_query:
            assistant_content = []
            for content in response.content:
                
                if content.type =='text':
                    print(f"\n---Text Content of Response:\n{content.text}\n---\n")
                    assistant_content.append(content)
                    # Checks if the entire response was just this single text block.
                    # this is the final answer and sets the flag to False to exit the while loop.
                    if(len(response.content) == 1):
                        
                        process_query= False
                # Checks if this part of the response is a request to use a tool.
                elif content.type == 'tool_use':
                
                    # Adds the tool use request to the list of the assistant's actions.
                    assistant_content.append(content)
                    # Appends the assistant's entire turn (the request to use a tool) to the main conversation history.
                    messages.append({'role':'assistant', 'content':assistant_content})
                    tool_id = content.id
                    tool_args = content.input
                    tool_name = content.name
                    
    
                    print(f"\n---Calling tool---\n Tool Name: {tool_name}\n Tool Args: {tool_args}\n---\n")
                    
                                       
                    # Call a tool
                    # Looks up the correct tool-handling object based on the tool's name. 
                    session = self.tool_to_session[tool_name] # new
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    
                    messages.append({"role": "user", 
                                      "content": [
                                          {
                                              "type": "tool_result",
                                              "tool_use_id":tool_id,
                                              "content": result.content
                                          }
                                      ]
                                    })
                    # This is the second (or subsequent) API call. The code sends the updated conversation history.
                    response = self.anthropic.messages.create(max_tokens = 2024,
                                      model = 'claude-3-7-sonnet-20250219', 
                                      tools = self.available_tools,
                                      messages = messages) 
                    
                    #  this code checks if the new response is a final, simple text answer.
                    if(len(response.content) == 1 and response.content[0].type == "text"):
                        # print(f"\n---Final Answer---\n{response.content[0].text}")
                        
                        process_query= False
        
        return f"\n---Final Answer---\n{response.content[0].text}"
        
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                # await self.process_query(query)
                res = await self.process_query(query)
                print(f"\n {res} \n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self): # new
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    # await chatbot.connect_to_servers()


    try:
        # the mcp clients and sessions are not initialized using "with"
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers() # new! 
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup() #new!
        print("\n Exiting the chatbot. \n") 
        


if __name__ == "__main__":
    asyncio.run(main())

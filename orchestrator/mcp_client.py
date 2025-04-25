import asyncio
import json
import subprocess
import logging
from typing import Dict, Any, Optional, List, Set

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_client")


class MCPClient:
    """Client for communicating with MCP servers"""

    def __init__(self, server_name: str, server_process: Optional[subprocess.Popen] = None):
        self.name = server_name
        self.process = server_process
        self.request_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.running = False
        self.task = None
        self.available_tools: Set[str] = set()  # Track available tools

    async def start(self):
        """Start the client and communication tasks"""
        self.running = True
        self.task = asyncio.create_task(self._process_queue())

        # Discover available tools
        await self._discover_tools()

    async def stop(self):
        """Stop the client and kill server process if needed"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server

        Args:
            tool_name: The name of the tool to call
            parameters: Parameters to pass to the tool

        Returns:
            Tool execution result
        """
        if tool_name not in self.available_tools:
            logger.warning(f"Tool '{tool_name}' not available in server '{self.name}'")
            return None

        request = {
            "version": "v1",
            "tool_calls": [
                {
                    "name": tool_name,
                    "parameters": parameters
                }
            ]
        }

        await self.request_queue.put(json.dumps(request))
        response_text = await self.response_queue.get()

        try:
            response = json.loads(response_text)
            if "results" in response and len(response["results"]) > 0:
                return response["results"][0]["content"]
            return None
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON response: {response_text}")
            return None

    async def _discover_tools(self):
        """Discover available tools on the MCP server"""
        # We'll implement a simple ping/list tools protocol
        # Send a request with a special tool name to get available tools
        logger.info(f"Discovering tools for {self.name}")

        try:
            if not self.process:
                logger.warning(f"No process for {self.name}, can't discover tools")
                return

            # Send a special request to list tools
            # In a real implementation, you'd use the MCP protocol's tool discovery capability
            discovery_request = {
                "version": "v1",
                "command": "discover_tools"
            }

            await self.request_queue.put(json.dumps(discovery_request))
            response_text = await self.response_queue.get()

            try:
                response = json.loads(response_text)
                if "tools" in response:
                    self.available_tools = set(tool["name"] for tool in response["tools"])
                    logger.info(f"Discovered tools for {self.name}: {self.available_tools}")
                else:
                    # For testing, populate with some expected tools based on server name
                    if self.name == "sql":
                        self.available_tools = {"Execute_SQL_Query", "Test_Connection", "Get_OMOP_Schema"}
                    elif self.name == "ollama":
                        self.available_tools = {"Generate_SQL", "Generate_Explanation", "Generate_Answer",
                                                "List_Available_Models"}
                    elif self.name == "validation":
                        self.available_tools = {"Validate_SQL_Query", "External_Validator", "Comprehensive_Validation"}
                    elif self.name == "agent":
                        self.available_tools = {"Get_Agent_Insights", "Get_Available_Agents", "Agent_Health_Check"}

                    logger.info(f"Using default tools for {self.name}: {self.available_tools}")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode tool discovery response: {response_text}")
                # Set default tools for testing
                if self.name == "sql":
                    self.available_tools = {"Execute_SQL_Query", "Test_Connection", "Get_OMOP_Schema"}
                elif self.name == "ollama":
                    self.available_tools = {"Generate_SQL", "Generate_Explanation", "Generate_Answer",
                                            "List_Available_Models"}
                elif self.name == "validation":
                    self.available_tools = {"Validate_SQL_Query", "External_Validator", "Comprehensive_Validation"}
                elif self.name == "agent":
                    self.available_tools = {"Get_Agent_Insights", "Get_Available_Agents", "Agent_Health_Check"}
        except Exception as e:
            logger.error(f"Error discovering tools for {self.name}: {e}")

    async def _process_queue(self):
        """Process the request queue and communicate with MCP server"""
        if not self.process:
            logger.error(f"No process for {self.name}, can't process queue")
            return

        while self.running:
            try:
                # Check if there's a request to process
                if not self.request_queue.empty():
                    request = await self.request_queue.get()

                    # Write request to process stdin
                    self.process.stdin.write(f"{request}\n".encode())
                    self.process.stdin.flush()

                    # Read response from process stdout
                    response_line = self.process.stdout.readline().decode().strip()
                    await self.response_queue.put(response_line)

                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in process_queue for {self.name}: {e}")
                await asyncio.sleep(1)  # Longer sleep on error
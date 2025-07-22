"""
Proper MCP (Model Context Protocol) integration for medical-a2a framework.
Implements standard JSON-RPC communication over stdio and SSE transports.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import json
import asyncio
import logging
import httpx
from httpx_sse import connect_sse
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class Transport(Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"
    
    @classmethod
    def from_url(cls, url: str) -> 'Transport':
        """Determine transport type from URL."""
        if url.startswith("stdio://"):
            return cls.STDIO
        elif url.startswith("http://") or url.startswith("https://"):
            return cls.SSE
        else:
            raise ValueError(f"Unknown transport type for URL: {url}")

@dataclass
class MCPServer:
    """MCP server configuration with support for multiple transports."""
    name: str
    url: str  # stdio://path/to/script.py or http://host:port
    description: str
    medical_speciality: Optional[str] = None
    
    # Additional config for stdio transport
    args: Optional[List[str]] = None  # Additional command line arguments
    env: Optional[Dict[str, str]] = None  # Environment variables
    working_dir: Optional[str] = None  # Working directory
    
    @property
    def transport(self) -> Transport:
        """Get transport type from URL."""
        return Transport.from_url(self.url)
    
    def get_command(self) -> List[str]:
        """Get command to execute for stdio transport."""
        if self.transport != Transport.STDIO:
            raise ValueError("Command only available for stdio transport")
            
        script_path = self.url.replace("stdio://", "")
        
        # Determine how to run the script
        if script_path.endswith(".py"):
            # For Python scripts, check if we should use uv
            base_cmd = ["uv", "run", "python"] if Path("pyproject.toml").exists() else ["python"]
            command = base_cmd + [script_path]
        else:
            # For executables
            command = [script_path]
            
        # Add any additional arguments
        if self.args:
            command.extend(self.args)
            
        return command

@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    medical_context: Optional[Dict[str, Any]] = None

class MCPConnection:
    """Base class for MCP connections."""
    
    def __init__(self, server: MCPServer):
        self.server = server
        self.request_id = 0
        self._initialized = False
        
    async def start(self):
        """Start the connection."""
        raise NotImplementedError
        
    async def stop(self):
        """Stop the connection."""
        raise NotImplementedError
        
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send JSON-RPC request."""
        raise NotImplementedError

class StdioMCPConnection(MCPConnection):
    """MCP connection over stdio transport."""
    
    def __init__(self, server: MCPServer):
        super().__init__(server)
        self.process: Optional[subprocess.Popen] = None
        self._response_futures: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the MCP server subprocess."""
        command = self.server.get_command()
        logger.info(f"Starting MCP server '{self.server.name}' with command: {command}")
        
        # Prepare environment
        env = None
        if self.server.env:
            import os
            env = os.environ.copy()
            env.update(self.server.env)
        
        self.process = subprocess.Popen(
            command,
            cwd=self.server.working_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Start async reader
        self._reader_task = asyncio.create_task(self._read_responses())
        
        # Give the process a moment to start
        await asyncio.sleep(0.5)
        
        # Check if process is still running
        if self.process.poll() is not None:
            stderr = self.process.stderr.read()
            raise RuntimeError(f"MCP server '{self.server.name}' failed to start: {stderr}")
        
        # Initialize connection
        await self._initialize()
        
    async def _read_responses(self):
        """Read responses from stdout."""
        while self.process and self.process.poll() is None:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, self.process.stdout.readline
                )
                
                if not line:
                    break
                    
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Try to parse as JSON
                if line.startswith("{") or line.startswith("["):
                    try:
                        response = json.loads(line)
                        await self._handle_response(response)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from {self.server.name}: {line}")
                else:
                    # Log non-JSON output for debugging
                    logger.debug(f"Non-JSON output from {self.server.name}: {line}")
                        
            except Exception as e:
                logger.error(f"Error reading from {self.server.name}: {e}")
                break
                
    async def _handle_response(self, response: Dict[str, Any]):
        """Handle JSON-RPC response."""
        request_id = response.get("id")
        
        if request_id and request_id in self._response_futures:
            future = self._response_futures.pop(request_id)
            if not future.done():
                future.set_result(response)
        else:
            # Handle notifications
            logger.debug(f"Notification from {self.server.name}: {response}")
            
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send JSON-RPC request."""
        self.request_id += 1
        request_id = self.request_id
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        # Create future for response
        future = asyncio.Future()
        self._response_futures[request_id] = future
        
        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str)
        self.process.stdin.flush()
        
        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            
            if "error" in response:
                error = response["error"]
                raise RuntimeError(f"MCP error from {self.server.name}: {error.get('message', 'Unknown error')}")
                
            return response.get("result", {})
            
        except asyncio.TimeoutError:
            self._response_futures.pop(request_id, None)
            raise RuntimeError(f"Request timeout for {method} on {self.server.name}")
            
    async def _initialize(self):
        """Initialize MCP connection."""
        result = await self.send_request("initialize", {
            "protocolVersion": "0.1.0",
            "capabilities": {},
            "clientInfo": {
                "name": "medical-a2a",
                "version": "1.0.0"
            }
        })
        
        self._initialized = True
        logger.info(f"Initialized connection to {self.server.name}")
        
    async def stop(self):
        """Stop the subprocess."""
        if self._reader_task:
            self._reader_task.cancel()
            
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

class SSEMCPConnection(MCPConnection):
    """MCP connection over SSE transport."""
    
    def __init__(self, server: MCPServer):
        super().__init__(server)
        self.client = httpx.AsyncClient(timeout=30.0)
        self._sse_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Connect to SSE endpoint."""
        # Initialize with a simple HTTP request first
        await self._initialize()
        
        # Start SSE listener
        self._sse_task = asyncio.create_task(self._listen_sse())
        
    async def _initialize(self):
        """Initialize connection."""
        # For SSE, we might need a different initialization
        # This depends on the server implementation
        self._initialized = True
        logger.info(f"Connected to SSE endpoint {self.server.name}")
        
    async def _listen_sse(self):
        """Listen for SSE events."""
        try:
            async with connect_sse(self.client, "GET", f"{self.server.url}/events") as event_source:
                async for event in event_source.aiter_sse():
                    logger.debug(f"SSE event from {self.server.name}: {event}")
        except Exception as e:
            logger.error(f"SSE error from {self.server.name}: {e}")
            
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send request via HTTP POST."""
        # For SSE transport, we send requests via HTTP POST
        # and receive responses via SSE or direct HTTP response
        response = await self.client.post(
            f"{self.server.url}/rpc",
            json={
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params or {}
            }
        )
        
        self.request_id += 1
        
        if response.status_code != 200:
            raise RuntimeError(f"HTTP error from {self.server.name}: {response.status_code}")
            
        result = response.json()
        
        if "error" in result:
            error = result["error"]
            raise RuntimeError(f"MCP error from {self.server.name}: {error.get('message', 'Unknown error')}")
            
        return result.get("result", {})
        
    async def stop(self):
        """Close connections."""
        if self._sse_task:
            self._sse_task.cancel()
            
        await self.client.aclose()

class MCPClient:
    """Client for a single MCP server."""
    
    def __init__(self, server: MCPServer):
        self.server = server
        self.connection: Optional[MCPConnection] = None
        self.tools: Dict[str, MCPTool] = {}
        
    async def connect(self):
        """Connect to the MCP server."""
        # Create appropriate connection based on transport
        if self.server.transport == Transport.STDIO:
            self.connection = StdioMCPConnection(self.server)
        else:
            self.connection = SSEMCPConnection(self.server)
            
        await self.connection.start()
        
        # Discover tools
        await self._discover_tools()
        
    async def _discover_tools(self):
        """Discover available tools."""
        # MCP uses 'tools/list' method
        result = await self.connection.send_request("tools/list")
        
        tools = result.get("tools", [])
        self.tools.clear()
        
        for tool_data in tools:
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema"),
                medical_context=tool_data.get("medicalContext")
            )
            self.tools[tool.name] = tool
            
        logger.info(f"Discovered {len(self.tools)} tools from {self.server.name}")
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in {self.server.name}")
            
        result = await self.connection.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        return result
        
    async def disconnect(self):
        """Disconnect from the server."""
        if self.connection:
            await self.connection.stop()
            self.connection = None
            self.tools.clear()

class MCPManager:
    """Manages multiple MCP server connections with proper protocol support."""
    
    def __init__(self, initial_servers: List[MCPServer]):
        self.servers: Dict[str, MCPServer] = {s.name: s for s in initial_servers}
        self.clients: Dict[str, MCPClient] = {}
        self.available_tools: Dict[str, Dict] = {}
        self._tool_cache_ttl = 300  # 5 minutes
        self._last_discovery = 0
        
    async def discover_servers(self, discovery_endpoint: Optional[str] = None):
        """Discover MCP servers from a registry."""
        if discovery_endpoint:
            async with httpx.AsyncClient() as client:
                response = await client.get(discovery_endpoint)
                servers_data = response.json()
                
                for server_data in servers_data:
                    server = MCPServer(**server_data)
                    self.servers[server.name] = server
                    
        # Connect to all servers
        await self._connect_all()
        
    async def register_server(self, server: MCPServer):
        """Register and connect to a new MCP server."""
        self.servers[server.name] = server
        await self._connect_server(server)
        
    async def _connect_all(self):
        """Connect to all registered servers."""
        tasks = [self._connect_server(server) for server in self.servers.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _connect_server(self, server: MCPServer):
        """Connect to a specific server."""
        try:
            # Disconnect existing client if any
            if server.name in self.clients:
                await self.clients[server.name].disconnect()
                
            # Create and connect new client
            client = MCPClient(server)
            await client.connect()
            self.clients[server.name] = client
            
            # Update available tools
            self._update_tool_registry(server.name, client)
            
        except Exception as e:
            logger.error(f"Failed to connect to {server.name}: {e}")
            
    def _update_tool_registry(self, server_name: str, client: MCPClient):
        """Update the tool registry with tools from a client."""
        # Remove old tools from this server
        self.available_tools = {
            k: v for k, v in self.available_tools.items() 
            if v["server"] != server_name
        }
        
        # Add new tools
        for tool_name, tool in client.tools.items():
            tool_id = f"{server_name}:{tool_name}"
            self.available_tools[tool_id] = {
                "server": server_name,
                "name": tool_name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "medical_context": tool.medical_context
            }
            
    async def get_available_tools(self) -> str:
        """Get formatted list of available tools."""
        tools_list = []
        for tool_id, tool_info in self.available_tools.items():
            params_str = json.dumps(tool_info.get("parameters", {}), indent=2)
            tools_list.append(
                f"- {tool_id}: {tool_info['description']}\n"
                f"  Parameters: {params_str}"
            )
            
        return "\n".join(tools_list)
        
    async def call_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool using format 'server_name:tool_name'."""
        parts = tool_id.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid tool_id format: '{tool_id}'. Use 'server_name:tool_name'")
            
        server_name, tool_name = parts
        
        if server_name not in self.clients:
            raise ValueError(f"Server '{server_name}' not connected")
            
        result = await self.clients[server_name].call_tool(tool_name, parameters)
        
        # Wrap result in expected format for backward compatibility
        return {"result": result}
        
    async def shutdown(self):
        """Disconnect all servers."""
        tasks = [client.disconnect() for client in self.clients.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.clients.clear()
        self.available_tools.clear()

class MCPDiscoveryMixin:
    """
    Mixin to add proper MCP support to medical agents.
    Compatible with existing framework while using standard MCP protocol.
    """
    
    def __init__(self, *args, mcp_servers: Optional[List[MCPServer]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcp_manager = MCPManager(mcp_servers or [])
        
        # Start connections in background
        asyncio.create_task(self._init_mcp_connections())
        
    async def _init_mcp_connections(self):
        """Initialize MCP connections."""
        try:
            await self.mcp_manager._connect_all()
        except Exception as e:
            logger.error(f"Failed to initialize MCP connections: {e}")
            
    async def discover_mcp_servers(self, discovery_endpoint: Optional[str] = None):
        """Discover available MCP servers."""
        await self.mcp_manager.discover_servers(discovery_endpoint)
        
    async def register_mcp_server(self, server: MCPServer):
        """Register a new MCP server."""
        await self.mcp_manager.register_server(server)
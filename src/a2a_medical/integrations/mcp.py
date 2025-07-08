from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import httpx
import asyncio

@dataclass
class MCPServer:
    """MCP server configuration."""
    name: str
    url: str
    description: str
    medical_speciality: Optional[str] = None

class MCPDiscoveryMixin:
    """Mixin to add MCP server discovery and tool calling to any medical agent."""

    def __init__(self, *args, mcp_servers: Optional[List[MCPServer]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcp_manager = MCPManager(mcp_servers or [])

    async def discover_mcp_servers(self, discovery_endpoint: Optional[str] = None):
        """Discover available MCP servers."""
        await self.mcp_manager.discover_servers(discovery_endpoint)

    async def register_mcp_server(self, server: MCPServer):
        """Register a new MCP server."""
        await self.mcp_manager.register_server(server)

class MCPManager:
    """Manages MCP server connections and tool discovery."""

    def __init__(self, initial_servers: List[MCPServer]):
        self.servers: Dict[str, MCPServer] = {s.name: s for s in initial_servers}
        self.available_tools: Dict[str, Dict] = {}
        self._tool_cache_ttl = 300  # 5 minutes
        self._last_discovery = 0

    async def discover_servers(self, discovery_endpoint: Optional[str] = None):
        """Discover MCP servers from a registry or local network."""
        if discovery_endpoint:
            async with httpx.AsyncClient() as client:
                response = await client.get(discovery_endpoint)
                servers_data = response.json()

                for server_data in servers_data:
                    server = MCPServer(**server_data)
                    self.servers[server.name] = server

        # Discover tools from all servers
        await self._refresh_tools()

    async def register_server(self, server: MCPServer):
        """Register a new MCP server."""
        self.servers[server.name] = server
        await self._refresh_server_tools(server)

    async def _refresh_tools(self):
        """Refresh available tools from all servers."""
        tasks = [self._refresh_server_tools(server) for server in self.servers.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _refresh_server_tools(self, server: MCPServer):
        """Get available tools from a specific MCP server."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{server.url}/tools")
                tools = response.json()

                for tool in tools:
                    tool_id = f"{server.name}:{tool['name']}"
                    self.available_tools[tool_id] = {
                        "server": server.name,
                        "server_url": server.url,
                        "name": tool['name'],
                        "description": tool.get('description', ''),
                        "parameters": tool.get('parameters', {}),
                        "medical_context": tool.get('medical_context', {})
                    }
        except Exception as e:
            print(f"⚠️ Failed to refresh tools from {server.name}: {e}")

    async def get_available_tools(self) -> str:
        """Get formatted list of available tools for LLM context."""
        # Refresh if cache is old
        import time
        if time.time() - self._last_discovery > self._tool_cache_ttl:
            await self._refresh_tools()
            self._last_discovery = time.time()

        tools_list = []
        for tool_id, tool_info in self.available_tools.items():
            tools_list.append(
                f"- {tool_id}: {tool_info['description']}\n"
                f"  Parameters: {tool_info['parameters']}"
            )

        return "\n".join(tools_list)

    async def call_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the appropriate MCP server."""
        if tool_id not in self.available_tools:
            raise ValueError(f"Tool {tool_id} not found")

        tool_info = self.available_tools[tool_id]
        server_url = tool_info['server_url']
        tool_name = tool_info['name']

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server_url}/tools/{tool_name}/call",
                json={"parameters": parameters}
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Tool call failed: {response.text}")
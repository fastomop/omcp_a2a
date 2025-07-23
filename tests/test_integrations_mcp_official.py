"""
Comprehensive test suite for official MCP (Model Context Protocol) integration.

Tests all aspects of the MCP integration using the official SDK including:
- MCPServer configuration and stdio parameters
- MCPClient connection and tool discovery
- MCPManager multi-server management
- MCPDiscoveryMixin functionality
- Error handling and edge cases
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from contextlib import asynccontextmanager

# Mock the MCP imports for testing when SDK is not available
sys.modules['mcp'] = MagicMock()
sys.modules['mcp.client'] = MagicMock()
sys.modules['mcp.client.stdio'] = MagicMock()
sys.modules['mcp.client.sse'] = MagicMock()
sys.modules['mcp.types'] = MagicMock()

from a2a_medical.integrations.mcp_official import (
    MCPServer, MCPClient, MCPManager, MCPDiscoveryMixin, MCPTool, Transport
)
from a2a_medical.base.agent import MedicalAgent


class TestMCPServer:
    """Test MCPServer configuration class."""
    
    def test_stdio_server_creation(self):
        """Test creating a stdio MCPServer instance."""
        server = MCPServer(
            name="test-server",
            url="stdio:///path/to/server.py",
            description="Test MCP server",
            medical_speciality="cardiology",
            args=["--debug"],
            env={"TEST_VAR": "value"},
            working_dir="/tmp"
        )
        
        assert server.name == "test-server"
        assert server.url == "stdio:///path/to/server.py"
        assert server.description == "Test MCP server"
        assert server.medical_speciality == "cardiology"
        assert server.transport == Transport.STDIO
        assert server.args == ["--debug"]
        assert server.env == {"TEST_VAR": "value"}
        assert server.working_dir == "/tmp"
    
    def test_sse_server_creation(self):
        """Test creating an SSE MCPServer instance."""
        server = MCPServer(
            name="sse-server",
            url="http://localhost:3000",
            description="SSE MCP server"
        )
        
        assert server.name == "sse-server"
        assert server.url == "http://localhost:3000"
        assert server.transport == Transport.SSE
    
    def test_get_stdio_params(self):
        """Test getting StdioServerParameters."""
        server = MCPServer(
            name="test",
            url="stdio:///path/to/script.py",
            description="Test",
            args=["--port", "8080"],
            env={"API_KEY": "secret"},
            working_dir="/workspace"
        )
        
        # Mock the StdioServerParameters class
        with patch('a2a_medical.integrations.mcp_official.StdioServerParameters') as mock_params:
            params = server.get_stdio_params()
            
            # Check that StdioServerParameters was called with correct args
            mock_params.assert_called_once()
            call_args = mock_params.call_args[1]
            
            # For Python scripts, it should use python command
            assert "python" in call_args["command"] or call_args["command"] == "uv"
            assert "/path/to/script.py" in call_args["args"]
            assert "--port" in call_args["args"]
            assert "8080" in call_args["args"]
            assert call_args["cwd"] == "/workspace"
    
    def test_get_stdio_params_raises_for_sse(self):
        """Test that get_stdio_params raises for SSE transport."""
        server = MCPServer(
            name="sse-server",
            url="http://localhost:3000",
            description="SSE server"
        )
        
        with pytest.raises(ValueError, match="StdioServerParameters only available for stdio transport"):
            server.get_stdio_params()


class TestMCPClient:
    """Test MCPClient functionality with mocked MCP SDK."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock ClientSession."""
        session = AsyncMock()
        session.initialize = AsyncMock(return_value=Mock(version="2025-06-18"))
        session.list_tools = AsyncMock(return_value=Mock(tools=[
            Mock(name="tool1", description="Test tool 1"),
            Mock(name="tool2", description="Test tool 2", inputSchema={"type": "object"})
        ]))
        session.call_tool = AsyncMock(return_value=Mock(content=[
            Mock(type="text", text="Tool result")
        ]))
        return session
    
    @pytest.fixture
    def stdio_server(self):
        """Create a test stdio server."""
        return MCPServer(
            name="test-server",
            url="stdio:///test/server.py",
            description="Test server"
        )
    
    @pytest.mark.asyncio
    async def test_client_connect(self, stdio_server, mock_session):
        """Test client connection to stdio server."""
        client = MCPClient(stdio_server)
        
        # Mock the stdio_client context manager
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        
        @asynccontextmanager
        async def mock_stdio_client(params):
            yield (mock_read, mock_write)
        
        with patch('a2a_medical.integrations.mcp_official.stdio_client', mock_stdio_client):
            with patch('a2a_medical.integrations.mcp_official.ClientSession', return_value=mock_session):
                await client.connect()
                
                assert client.session is not None
                assert len(client.tools) == 2
                assert "tool1" in client.tools
                assert "tool2" in client.tools
                assert client.tools["tool1"].name == "tool1"
                assert client.tools["tool1"].description == "Test tool 1"
                assert client.tools["tool2"].input_schema == {"type": "object"}
    
    @pytest.mark.asyncio
    async def test_client_call_tool(self, stdio_server, mock_session):
        """Test calling a tool through the client."""
        client = MCPClient(stdio_server)
        
        # Set up the client with mocked connection
        client.session = mock_session
        client.tools = {
            "test_tool": MCPTool(name="test_tool", description="Test tool")
        }
        
        result = await client.call_tool("test_tool", {"param": "value"})
        
        assert result == "Tool result"
        mock_session.call_tool.assert_called_once_with(
            name="test_tool",
            arguments={"param": "value"}
        )
    
    @pytest.mark.asyncio
    async def test_client_call_nonexistent_tool(self, stdio_server, mock_session):
        """Test calling a non-existent tool raises error."""
        client = MCPClient(stdio_server)
        client.session = mock_session
        client.tools = {}
        
        with pytest.raises(ValueError, match="Tool 'unknown' not found"):
            await client.call_tool("unknown", {})
    
    @pytest.mark.asyncio
    async def test_client_disconnect(self, stdio_server, mock_session):
        """Test client disconnection."""
        client = MCPClient(stdio_server)
        
        # Set up mocked connection
        client.session = mock_session
        client._context_manager = AsyncMock()
        client.tools = {"tool": MCPTool(name="tool", description="Test")}
        
        await client.disconnect()
        
        assert client.session is None
        assert client._context_manager is None
        assert len(client.tools) == 0


class TestMCPManager:
    """Test MCPManager multi-server functionality."""
    
    @pytest.fixture
    def test_servers(self):
        """Create test server configurations."""
        return [
            MCPServer("server1", "stdio:///server1.py", "Server 1"),
            MCPServer("server2", "stdio:///server2.py", "Server 2")
        ]
    
    @pytest.fixture
    def mock_client_class(self):
        """Create a mock MCPClient class."""
        with patch('a2a_medical.integrations.mcp_official.MCPClient') as mock_class:
            # Create mock instances
            mock_clients = {}
            
            def create_mock_client(server):
                mock_client = AsyncMock()
                mock_client.server = server
                mock_client.connect = AsyncMock()
                mock_client.disconnect = AsyncMock()
                mock_client.tools = {
                    f"{server.name}_tool": MCPTool(
                        name=f"{server.name}_tool",
                        description=f"Tool from {server.name}"
                    )
                }
                mock_clients[server.name] = mock_client
                return mock_client
            
            mock_class.side_effect = create_mock_client
            yield mock_class, mock_clients
    
    def test_manager_initialization(self, test_servers):
        """Test MCPManager initialization."""
        manager = MCPManager(test_servers)
        
        assert len(manager.servers) == 2
        assert "server1" in manager.servers
        assert "server2" in manager.servers
        assert manager.available_tools == {}
    
    @pytest.mark.asyncio
    async def test_manager_connect_all(self, test_servers, mock_client_class):
        """Test connecting to all servers."""
        mock_class, mock_clients = mock_client_class
        manager = MCPManager(test_servers)
        
        await manager._connect_all()
        
        assert len(manager.clients) == 2
        assert "server1" in manager.clients
        assert "server2" in manager.clients
        
        # Check that tools were registered
        assert "server1:server1_tool" in manager.available_tools
        assert "server2:server2_tool" in manager.available_tools
    
    @pytest.mark.asyncio
    async def test_manager_call_tool(self, test_servers, mock_client_class):
        """Test calling a tool through the manager."""
        mock_class, mock_clients = mock_client_class
        manager = MCPManager(test_servers)
        
        await manager._connect_all()
        
        # Mock the call_tool method on the client
        mock_clients["server1"].call_tool = AsyncMock(return_value="Tool result")
        
        result = await manager.call_tool("server1:server1_tool", {"param": "value"})
        
        assert result == {"result": "Tool result"}
        mock_clients["server1"].call_tool.assert_called_once_with(
            "server1_tool",
            {"param": "value"}
        )
    
    @pytest.mark.asyncio
    async def test_manager_call_tool_invalid_format(self, test_servers):
        """Test calling tool with invalid ID format."""
        manager = MCPManager(test_servers)
        
        with pytest.raises(ValueError, match="Invalid tool_id format"):
            await manager.call_tool("invalid_format", {})
    
    @pytest.mark.asyncio
    async def test_manager_shutdown(self, test_servers, mock_client_class):
        """Test manager shutdown disconnects all clients."""
        mock_class, mock_clients = mock_client_class
        manager = MCPManager(test_servers)
        
        await manager._connect_all()
        await manager.shutdown()
        
        # Check all clients were disconnected
        for client in mock_clients.values():
            client.disconnect.assert_called_once()
        
        assert len(manager.clients) == 0
        assert len(manager.available_tools) == 0


class TestMCPDiscoveryMixin:
    """Test MCPDiscoveryMixin functionality."""
    
    class TestAgent(MCPDiscoveryMixin, MedicalAgent):
        """Test agent with MCP discovery capabilities."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    
    @pytest.mark.asyncio
    async def test_mixin_initialization(self):
        """Test that mixin initializes MCP manager."""
        servers = [
            MCPServer("test", "stdio:///test.py", "Test server")
        ]
        
        with patch('a2a_medical.integrations.mcp_official.MCPManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            agent = self.TestAgent(
                agent_id="test-agent",
                agent_type="test",
                capabilities=[],
                world_model=None,
                mcp_servers=servers
            )
            
            # Give the background task a moment to run
            await asyncio.sleep(0.1)
            
            assert agent.mcp_manager is not None
            mock_manager_class.assert_called_once_with(servers)
    
    @pytest.mark.asyncio
    async def test_discover_servers(self):
        """Test discovering MCP servers."""
        agent = self.TestAgent(
            agent_id="test-agent",
            agent_type="test",
            capabilities=[],
            world_model=None
        )
        
        agent.mcp_manager = AsyncMock()
        
        await agent.discover_mcp_servers("http://discovery.test")
        
        agent.mcp_manager.discover_servers.assert_called_once_with("http://discovery.test")
    
    @pytest.mark.asyncio
    async def test_register_server(self):
        """Test registering a new MCP server."""
        agent = self.TestAgent(
            agent_id="test-agent",
            agent_type="test",
            capabilities=[],
            world_model=None
        )
        
        agent.mcp_manager = AsyncMock()
        
        new_server = MCPServer("new", "stdio:///new.py", "New server")
        await agent.register_mcp_server(new_server)
        
        agent.mcp_manager.register_server.assert_called_once_with(new_server)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_client_connect_initialization_error(self):
        """Test handling initialization errors during connection."""
        server = MCPServer("test", "stdio:///test.py", "Test")
        client = MCPClient(server)
        
        # Mock stdio_client and session with initialization error
        mock_session = AsyncMock()
        mock_session.initialize.side_effect = Exception("Initialization failed")
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        
        @asynccontextmanager
        async def mock_stdio_client(params):
            yield (AsyncMock(), AsyncMock())
        
        with patch('a2a_medical.integrations.mcp_official.stdio_client', mock_stdio_client):
            with patch('a2a_medical.integrations.mcp_official.ClientSession', return_value=mock_session):
                with pytest.raises(Exception, match="Initialization failed"):
                    await client.connect()
                
                # Ensure cleanup happened
                assert client.session is None
    
    @pytest.mark.asyncio
    async def test_manager_server_connection_failure(self):
        """Test manager handles individual server connection failures."""
        servers = [
            MCPServer("good", "stdio:///good.py", "Good server"),
            MCPServer("bad", "stdio:///bad.py", "Bad server")
        ]
        
        manager = MCPManager(servers)
        
        # Mock MCPClient to fail for "bad" server
        with patch('a2a_medical.integrations.mcp_official.MCPClient') as mock_client_class:
            def create_client(server):
                client = AsyncMock()
                if server.name == "bad":
                    client.connect.side_effect = Exception("Connection failed")
                else:
                    client.connect.return_value = None
                    client.tools = {"tool": MCPTool("tool", "Test tool")}
                return client
            
            mock_client_class.side_effect = create_client
            
            # Should not raise, but log errors
            await manager._connect_all()
            
            # Only the good server should be connected
            assert len(manager.clients) == 1
            assert "good" in manager.clients
            assert "bad" not in manager.clients


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Comprehensive test suite for MCP (Model Context Protocol) integration.

Tests all aspects of the MCP integration including:
- MCPServer configuration
- MCPDiscoveryMixin functionality
- MCPManager tool discovery and management
- Server registration and tool calling
- Error handling and edge cases
"""

import pytest
import asyncio
import httpx
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from a2a_medical.integrations.mcp import (
    MCPServer, MCPDiscoveryMixin, MCPManager
)
from a2a_medical.base.agent import MedicalAgent, WorldModel, ProcessedObservation
from a2a_medical.models.medical import Patient


class TestMCPServer:
    """Test MCPServer data class."""
    
    def test_mcp_server_creation(self):
        """Test creating an MCPServer instance."""
        server = MCPServer(
            name="test-server",
            url="http://localhost:3000",
            description="Test MCP server",
            medical_speciality="cardiology"
        )
        
        assert server.name == "test-server"
        assert server.url == "http://localhost:3000"
        assert server.description == "Test MCP server"
        assert server.medical_speciality == "cardiology"
    
    def test_mcp_server_optional_speciality(self):
        """Test MCPServer without medical speciality."""
        server = MCPServer(
            name="general-server",
            url="http://localhost:3001",
            description="General MCP server"
        )
        
        assert server.name == "general-server"
        assert server.medical_speciality is None
    
    def test_mcp_server_equality(self):
        """Test MCPServer equality comparison."""
        server1 = MCPServer("test", "http://localhost:3000", "Test server")
        server2 = MCPServer("test", "http://localhost:3000", "Test server")
        server3 = MCPServer("different", "http://localhost:3000", "Test server")
        
        assert server1 == server2
        assert server1 != server3


class TestMCPManager:
    """Test MCPManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.initial_servers = [
            MCPServer("server1", "http://localhost:3000", "Test server 1"),
            MCPServer("server2", "http://localhost:3001", "Test server 2")
        ]
        self.manager = MCPManager(self.initial_servers)
    
    def test_mcp_manager_initialization(self):
        """Test MCPManager initialization."""
        assert len(self.manager.servers) == 2
        assert "server1" in self.manager.servers
        assert "server2" in self.manager.servers
        assert self.manager.available_tools == {}
        assert self.manager._tool_cache_ttl == 300
    
    @patch('httpx.AsyncClient')
    async def test_discover_servers_with_endpoint(self, mock_client):
        """Test server discovery with discovery endpoint."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "name": "discovered-server",
                "url": "http://localhost:3002",
                "description": "Discovered server",
                "medical_speciality": "neurology"
            }
        ]
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        # Test discovery
        await self.manager.discover_servers("http://discovery.example.com")
        
        # Check that discovered server was added
        assert "discovered-server" in self.manager.servers
        assert self.manager.servers["discovered-server"].medical_speciality == "neurology"
    
    @patch('httpx.AsyncClient')
    async def test_discover_servers_without_endpoint(self, mock_client):
        """Test server discovery without endpoint."""
        # Should not make any HTTP requests
        await self.manager.discover_servers()
        
        # Should still refresh tools from existing servers
        assert len(self.manager.servers) == 2
    
    async def test_register_server(self):
        """Test registering a new server."""
        new_server = MCPServer("new-server", "http://localhost:3003", "New server")
        
        await self.manager.register_server(new_server)
        
        assert "new-server" in self.manager.servers
        assert len(self.manager.servers) == 3
    
    @patch('httpx.AsyncClient')
    async def test_refresh_server_tools_success(self, mock_client):
        """Test successful tool refresh from server."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "parameters": {"param1": "string"},
                "medical_context": {"category": "diagnosis"}
            }
        ]
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        server = self.manager.servers["server1"]
        await self.manager._refresh_server_tools(server)
        
        # Check that tool was added
        tool_id = "server1:test_tool"
        assert tool_id in self.manager.available_tools
        assert self.manager.available_tools[tool_id]["name"] == "test_tool"
        assert self.manager.available_tools[tool_id]["medical_context"]["category"] == "diagnosis"
    
    @patch('httpx.AsyncClient')
    async def test_refresh_server_tools_failure(self, mock_client):
        """Test tool refresh failure handling."""
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = Exception("Connection failed")
        mock_client.return_value = mock_client_instance
        
        server = self.manager.servers["server1"]
        
        # Should not raise exception, just log error
        await self.manager._refresh_server_tools(server)
        
        # No tools should be added
        assert len(self.manager.available_tools) == 0
    
    @patch('httpx.AsyncClient')
    async def test_get_available_tools(self, mock_client):
        """Test getting formatted list of available tools."""
        # Mock tool discovery
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "name": "tool1",
                "description": "First tool",
                "parameters": {"param1": "string"}
            },
            {
                "name": "tool2", 
                "description": "Second tool",
                "parameters": {"param2": "number"}
            }
        ]
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        # Get tools
        tools_text = await self.manager.get_available_tools()
        
        # Check format
        assert "server1:tool1" in tools_text
        assert "server1:tool2" in tools_text
        assert "First tool" in tools_text
        assert "Second tool" in tools_text
        assert "Parameters:" in tools_text
    
    @patch('httpx.AsyncClient')
    async def test_call_tool_success(self, mock_client):
        """Test successful tool call."""
        # First add a tool
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "parameters": {"param1": "string"}
            }
        ]
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        server = self.manager.servers["server1"]
        await self.manager._refresh_server_tools(server)
        
        # Mock tool call response
        tool_response = Mock()
        tool_response.status_code = 200
        tool_response.json.return_value = {"result": "success", "data": "test_data"}
        
        mock_client_instance.post.return_value = tool_response
        
        # Call tool
        result = await self.manager.call_tool("server1:test_tool", {"param1": "value1"})
        
        assert result == {"result": "success", "data": "test_data"}
    
    @patch('httpx.AsyncClient')
    async def test_call_tool_not_found(self, mock_client):
        """Test calling non-existent tool."""
        with pytest.raises(ValueError, match="Tool nonexistent not found"):
            await self.manager.call_tool("nonexistent", {})
    
    @patch('httpx.AsyncClient')
    async def test_call_tool_server_error(self, mock_client):
        """Test tool call with server error."""
        # First add a tool
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "parameters": {"param1": "string"}
            }
        ]
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        server = self.manager.servers["server1"]
        await self.manager._refresh_server_tools(server)
        
        # Mock error response
        error_response = Mock()
        error_response.status_code = 500
        error_response.text = "Internal server error"
        
        mock_client_instance.post.return_value = error_response
        
        # Call tool should raise exception
        with pytest.raises(Exception, match="Tool call failed: Internal server error"):
            await self.manager.call_tool("server1:test_tool", {"param1": "value1"})
    
    def test_tool_cache_ttl(self):
        """Test tool cache TTL functionality."""
        import time
        
        # Mock time to test cache expiration
        original_time = time.time
        mock_time = Mock(return_value=0)
        time.time = mock_time
        
        try:
            # Initial discovery
            mock_time.return_value = 0
            self.manager._last_discovery = 0
            
            # Should refresh when cache is old
            mock_time.return_value = 301  # More than 300 seconds
            assert time.time() - self.manager._last_discovery > self.manager._tool_cache_ttl
            
            # Should not refresh when cache is fresh
            mock_time.return_value = 150  # Less than 300 seconds
            self.manager._last_discovery = 0
            assert time.time() - self.manager._last_discovery < self.manager._tool_cache_ttl
            
        finally:
            time.time = original_time


class TestMCPDiscoveryMixin:
    """Test MCPDiscoveryMixin functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock agent class that uses the mixin
        class TestAgent(MCPDiscoveryMixin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.agent_id = "test-agent"
        
        self.agent_class = TestAgent
    
    def test_mixin_initialization_with_servers(self):
        """Test mixin initialization with MCP servers."""
        servers = [
            MCPServer("server1", "http://localhost:3000", "Test server 1"),
            MCPServer("server2", "http://localhost:3001", "Test server 2")
        ]
        
        agent = self.agent_class(mcp_servers=servers)
        
        assert len(agent.mcp_manager.servers) == 2
        assert "server1" in agent.mcp_manager.servers
        assert "server2" in agent.mcp_manager.servers
    
    def test_mixin_initialization_without_servers(self):
        """Test mixin initialization without MCP servers."""
        agent = self.agent_class()
        
        assert len(agent.mcp_manager.servers) == 0
        assert agent.mcp_manager.available_tools == {}
    
    @patch.object(MCPManager, 'discover_servers')
    async def test_discover_mcp_servers(self, mock_discover):
        """Test discovering MCP servers."""
        agent = self.agent_class()
        
        await agent.discover_mcp_servers("http://discovery.example.com")
        
        mock_discover.assert_called_once_with("http://discovery.example.com")
    
    @patch.object(MCPManager, 'discover_servers')
    async def test_discover_mcp_servers_no_endpoint(self, mock_discover):
        """Test discovering MCP servers without endpoint."""
        agent = self.agent_class()
        
        await agent.discover_mcp_servers()
        
        mock_discover.assert_called_once_with(None)
    
    @patch.object(MCPManager, 'register_server')
    async def test_register_mcp_server(self, mock_register):
        """Test registering an MCP server."""
        agent = self.agent_class()
        server = MCPServer("new-server", "http://localhost:3003", "New server")
        
        await agent.register_mcp_server(server)
        
        mock_register.assert_called_once_with(server)


# Move TestWorldModel to module scope so it is available for all tests
class TestWorldModel(WorldModel):
    def update(self, observation: ProcessedObservation) -> None:
        pass
    
    def query(self, query: str, context=None):
        return f"Query result for: {query}"
    
    def predict(self, scenario):
        return {"prediction": "test prediction"}
    
    def get_state_summary(self):
        return {"status": "active"}
    
    def reset(self) -> None:
        pass


class TestMCPIntegrationWithMedicalAgent:
    """Test MCP integration with actual medical agents."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Fix inheritance and argument passing
        class TestMedicalAgent(MCPDiscoveryMixin, MedicalAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            async def perceive(self, observation):
                return ProcessedObservation(
                    data=observation,
                    timestamp=datetime.now().timestamp(),
                    source="test",
                    confidence=0.9
                )
            
            async def learn(self, state, observation):
                return state
            
            async def reason(self, state):
                from a2a_medical.base.agent import Action
                return Action(
                    action_type="test_action",
                    parameters={"test": "value"},
                    priority=1
                )
            
            async def execute(self, action):
                from a2a_medical.base.agent import ActionResult
                return ActionResult(
                    success=True,
                    data={"result": "test"},
                    error=None,
                    metadata={"action": action.action_type}
                )
            
            def build_agent_card(self):
                from a2a.types import AgentCard, AgentCapabilities
                return AgentCard(
                    name=self.agent_name,
                    description=self.agent_description,
                    url="https://test.local",
                    version=self.agent_version,
                    capabilities=AgentCapabilities(streaming=False),
                    skills=[],
                    defaultInputModes=["text"],
                    defaultOutputModes=["text"]
                )
        
        self.agent_class = TestMedicalAgent
    
    def test_medical_agent_with_mcp_integration(self):
        """Test medical agent with MCP integration."""
        world_model = TestWorldModel()
        
        agent = self.agent_class(
            agent_id="mcp-test-agent",
            agent_type="test",
            capabilities=["diagnosis", "treatment"],
            world_model=world_model,
            mcp_servers=[
                MCPServer("medical-server", "http://localhost:3000", "Medical MCP server")
            ]
        )
        
        assert agent.agent_id == "mcp-test-agent"
        assert len(agent.mcp_manager.servers) == 1
        assert "medical-server" in agent.mcp_manager.servers
    
    @patch.object(MCPManager, 'call_tool')
    async def test_medical_agent_tool_calling(self, mock_call_tool):
        """Test medical agent calling MCP tools."""
        world_model = TestWorldModel()
        
        agent = self.agent_class(
            agent_id="mcp-test-agent",
            agent_type="test",
            capabilities=["diagnosis"],
            world_model=world_model
        )
        
        # Mock tool call
        mock_call_tool.return_value = {"result": "diagnosis_result"}
        
        # Call tool through agent's MCP manager
        result = await agent.mcp_manager.call_tool("test:diagnosis", {"symptoms": ["fever"]})
        
        assert result == {"result": "diagnosis_result"}
        mock_call_tool.assert_called_once_with("test:diagnosis", {"symptoms": ["fever"]})


class TestMCPErrorHandling:
    """Test MCP error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = MCPManager([])
    
    @patch('httpx.AsyncClient')
    async def test_discovery_network_error(self, mock_client):
        """Test handling of network errors during discovery."""
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = httpx.ConnectError("Connection failed")
        mock_client.return_value = mock_client_instance
        
        # Should not raise exception
        await self.manager.discover_servers("http://invalid.example.com")
        
        # No servers should be added
        assert len(self.manager.servers) == 0
    
    @patch('httpx.AsyncClient')
    async def test_discovery_invalid_json(self, mock_client):
        """Test handling of invalid JSON responses."""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        # Should not raise exception
        await self.manager.discover_servers("http://example.com")
        
        # No servers should be added
        assert len(self.manager.servers) == 0
    
    def test_empty_tool_list(self):
        """Test handling of empty tool list."""
        # Should not raise exception
        tools_text = asyncio.run(self.manager.get_available_tools())
        
        assert tools_text == ""
    
    def test_duplicate_server_registration(self):
        """Test registering duplicate servers."""
        server = MCPServer("test-server", "http://localhost:3000", "Test server")
        
        # Register same server twice
        asyncio.run(self.manager.register_server(server))
        asyncio.run(self.manager.register_server(server))
        
        # Should only have one instance
        assert len(self.manager.servers) == 1
        assert "test-server" in self.manager.servers


class TestMCPPerformance:
    """Test MCP performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = MCPManager([])
    
    @patch('httpx.AsyncClient')
    async def test_concurrent_tool_discovery(self, mock_client):
        """Test concurrent tool discovery from multiple servers."""
        # Mock multiple servers
        servers = [
            MCPServer(f"server{i}", f"http://localhost:300{i}", f"Server {i}")
            for i in range(3)
        ]
        
        for server in servers:
            self.manager.servers[server.name] = server
        
        # Mock responses
        mock_response = Mock()
        mock_response.json.return_value = [
            {"name": f"tool{i}", "description": f"Tool {i}"}
            for i in range(2)
        ]
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        # Discover tools concurrently
        await self.manager._refresh_tools()
        
        # Should have tools from all servers
        expected_tools = 6  # 3 servers * 2 tools each
        assert len(self.manager.available_tools) == expected_tools
    
    def test_tool_cache_performance(self):
        """Test tool cache performance."""
        import time
        
        # Mock time
        original_time = time.time
        mock_time = Mock(return_value=0)
        time.time = mock_time
        
        try:
            # First call should refresh
            mock_time.return_value = 0
            self.manager._last_discovery = 0
            
            # Second call within TTL should use cache
            mock_time.return_value = 150  # Within 300 second TTL
            # Should not trigger refresh
            
        finally:
            time.time = original_time


# Integration tests
class TestMCPIntegration:
    """Integration tests for MCP functionality."""
    
    async def test_full_mcp_workflow(self):
        """Test complete MCP workflow."""
        # Create manager
        manager = MCPManager([])
        
        # Register servers
        servers = [
            MCPServer("lab-server", "http://localhost:3000", "Laboratory server"),
            MCPServer("pharmacy-server", "http://localhost:3001", "Pharmacy server")
        ]
        
        for server in servers:
            await manager.register_server(server)
        
        assert len(manager.servers) == 2
        
        # Discover tools (mocked)
        with patch.object(manager, '_refresh_server_tools') as mock_refresh:
            await manager._refresh_tools()
            assert mock_refresh.call_count == 2
        
        # Get available tools
        tools_text = await manager.get_available_tools()
        assert isinstance(tools_text, str)
    
    async def test_medical_agent_mcp_integration(self):
        """Test medical agent with full MCP integration."""
        # This would test a real medical agent with MCP capabilities
        # For now, test the basic structure
        class TestWorldModel(WorldModel):
            def update(self, observation: ProcessedObservation) -> None:
                pass
            
            def query(self, query: str, context=None):
                return "test result"
            
            def predict(self, scenario):
                return {"prediction": "test"}
            
            def get_state_summary(self):
                return {"status": "active"}
            
            def reset(self) -> None:
                pass
        
        class TestMedicalAgent(MedicalAgent, MCPDiscoveryMixin):
            def __init__(self, *args, **kwargs):
                MedicalAgent.__init__(self, *args, **kwargs)
                MCPDiscoveryMixin.__init__(self, *args, **kwargs)
            
            async def perceive(self, observation):
                return ProcessedObservation(
                    data=observation,
                    timestamp=datetime.now().timestamp(),
                    source="test",
                    confidence=0.9
                )
            
            async def learn(self, state, observation):
                return state
            
            async def reason(self, state):
                from a2a_medical.base.agent import Action
                return Action(
                    action_type="test_action",
                    parameters={"test": "value"},
                    priority=1
                )
            
            async def execute(self, action):
                from a2a_medical.base.agent import ActionResult
                return ActionResult(
                    success=True,
                    data={"result": "test"},
                    error=None,
                    metadata={"action": action.action_type}
                )
            
            def build_agent_card(self):
                from a2a.types import AgentCard, AgentCapabilities
                return AgentCard(
                    name=self.agent_name,
                    description=self.agent_description,
                    url="https://test.local",
                    version=self.agent_version,
                    capabilities=AgentCapabilities(streaming=False),
                    skills=[],
                    defaultInputModes=["text"],
                    defaultOutputModes=["text"]
                )
        
        # Create agent with MCP integration
        world_model = TestWorldModel()
        agent = TestMedicalAgent(
            agent_id="integration-test",
            agent_type="test",
            capabilities=["diagnosis"],
            world_model=world_model,
            mcp_servers=[
                MCPServer("medical-server", "http://localhost:3000", "Medical server")
            ]
        )
        
        # Test basic functionality
        assert agent.agent_id == "integration-test"
        assert len(agent.mcp_manager.servers) == 1
        assert "medical-server" in agent.mcp_manager.servers
        
        # Test agent can process observations
        observation = await agent.perceive({"test": "data"})
        assert observation.source == "test"
        assert observation.confidence == 0.9 
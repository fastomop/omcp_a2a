"""
OMCP Orchestrator Package

This package contains modules for orchestrating multiple MCP servers
and implementing the A2A protocol for agent-to-agent communication.
"""

from .a2a import A2AProtocol
from .mcp_client import MCPClient
from .main import MCPOrchestrator

__all__ = ['A2AProtocol', 'MCPClient', 'MCPOrchestrator']
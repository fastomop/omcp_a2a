"""
MCP (Model Context Protocol) integration for medical-a2a framework.

This module implements the standard MCP protocol with proper JSON-RPC 
communication over stdio and SSE transports.

Key features:
- Subprocess launching for stdio:// URLs
- HTTP/SSE support for remote servers
- Automatic tool discovery
- Proper lifecycle management
"""

# Export everything from the proper implementation
from .mcp_proper import (
    # Main classes
    MCPServer,
    MCPDiscoveryMixin,
    MCPManager,
    MCPTool,
    Transport,
    
    # Advanced usage (if needed)
    MCPClient,
    MCPConnection,
    StdioMCPConnection,
    SSEMCPConnection
)

__all__ = [
    'MCPServer',
    'MCPDiscoveryMixin', 
    'MCPManager',
    'MCPTool',
    'Transport',
    'MCPClient',
    'MCPConnection',
    'StdioMCPConnection',
    'SSEMCPConnection'
]
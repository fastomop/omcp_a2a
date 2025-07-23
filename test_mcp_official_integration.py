#!/usr/bin/env python3
"""
Simple integration test for the official MCP integration.
This script tests the new implementation against a real MCP server.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from a2a_medical.integrations.mcp_official import MCPServer, MCPManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_mcp_official_integration():
    """Test the official MCP integration with a real server."""
    print("🧪 Testing Official MCP Integration")
    print("=" * 50)
    
    # Check if MCP SDK is available
    try:
        import mcp
        print("✅ MCP SDK is installed")
    except ImportError:
        print("❌ MCP SDK not found. Install with: uv add mcp")
        return
    
    # Create a test server configuration
    # This assumes you have the OMCP server available
    omcp_path = "/Users/k24118093/Documents/omcp_server/src/omcp/main.py"
    if not Path(omcp_path).exists():
        print(f"⚠️  OMCP server not found at {omcp_path}")
        print("   Using a mock server configuration for testing")
        
        # Create a simple test server config
        test_server = MCPServer(
            name="test_server",
            url="stdio:///usr/bin/echo",  # Simple echo command for testing
            description="Test echo server",
            args=["Hello from MCP"]
        )
    else:
        # Use the actual OMCP server
        test_server = MCPServer(
            name="omop_db_server",
            url=f"stdio://{omcp_path}",
            description="OMOP CDM database access via MCP",
            working_dir="/Users/k24118093/Documents/omcp_server",
            env={
                "DB_TYPE": "duckdb",
                "DB_PATH": "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb",
                "CDM_SCHEMA": "cdm",
                "VOCAB_SCHEMA": "cdm"
            }
        )
    
    # Create MCP manager
    manager = MCPManager([test_server])
    
    try:
        # Test 1: Server Configuration
        print("\n1️⃣ Testing Server Configuration...")
        print(f"   Name: {test_server.name}")
        print(f"   URL: {test_server.url}")
        print(f"   Transport: {test_server.transport}")
        
        if test_server.transport.value == "stdio":
            stdio_params = test_server.get_stdio_params()
            print(f"   Command: {stdio_params.command}")
            print(f"   Args: {stdio_params.args}")
        
        # Test 2: Connection
        print("\n2️⃣ Testing Connection...")
        await manager._connect_all()
        print("✅ Connected successfully")
        
        # Test 3: Tool Discovery
        print("\n3️⃣ Testing Tool Discovery...")
        if test_server.name in manager.clients:
            client = manager.clients[test_server.name]
            print(f"   Found {len(client.tools)} tools:")
            for tool_name, tool in client.tools.items():
                print(f"   - {tool_name}: {tool.description}")
        
        # Test 4: Available Tools
        print("\n4️⃣ Testing Available Tools...")
        tools_list = await manager.get_available_tools()
        if tools_list:
            print("Available tools:")
            print(tools_list)
        else:
            print("   No tools available (this is normal for echo server)")
        
        # Test 5: Tool Invocation (if OMCP server is available)
        if test_server.name == "omop_db_server" and manager.available_tools:
            print("\n5️⃣ Testing Tool Invocation...")
            
            # Try to get schema information
            if "omop_db_server:Get_Information_Schema" in manager.available_tools:
                try:
                    result = await manager.call_tool("omop_db_server:Get_Information_Schema", {})
                    print("✅ Schema tool called successfully")
                    print(f"   Result type: {type(result)}")
                except Exception as e:
                    print(f"❌ Tool invocation failed: {e}")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n🧹 Shutting down...")
        await manager.shutdown()
        print("✅ Cleanup complete")

def main():
    """Run the integration test."""
    print("Starting MCP Official Integration Test")
    print("=====================================")
    
    try:
        asyncio.run(test_mcp_official_integration())
        print("\n✅ Integration test completed")
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
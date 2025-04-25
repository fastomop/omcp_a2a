from mcp.server.fastmcp import FastMCP
import httpx
from typing import Dict, Any, List, Optional
import json

# Initialize MCP server
mcp = FastMCP(name="OMOP Agent Integration MCP Server")


# A2A Protocol implementation
class A2AProtocol:
    """Implementation of the Agent-to-Agent (A2A) protocol"""

    @staticmethod
    def create_message(content: str, role: str = "user", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a message following A2A protocol

        Args:
            content: The message content
            role: The role of the sender (user, assistant, etc.)
            metadata: Optional metadata

        Returns:
            A2A formatted message
        """
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        return message

    @staticmethod
    def create_request(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create an A2A request

        Args:
            messages: List of messages
            tools: Optional list of available tools

        Returns:
            A2A formatted request
        """
        request = {
            "messages": messages
        }

        if tools:
            request["tools"] = tools

        return request

    @staticmethod
    async def send_request(url: str, request_data: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """Send a request to another agent using A2A protocol

        Args:
            url: The URL of the agent
            request_data: The A2A formatted request
            timeout: Request timeout in seconds

        Returns:
            Agent response
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=request_data,
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {
                "error": str(e),
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"Error communicating with agent: {str(e)}"
                    }
                ]
            }


# Initialize A2A protocol
a2a = A2AProtocol()


@mcp.tool(
    name="Get_Agent_Insights",
    description="Get insights from an external agent using A2A protocol"
)
async def get_agent_insights(prompt: str, sql: str, agent_type: str, agent_url: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get insights from an external agent using A2A protocol

    Args:
        prompt: Original natural language prompt
        sql: Generated SQL query
        agent_type: Type of agent to consult
        agent_url: Optional URL override for the agent
        context: Additional context for the agent

    Returns:
        Agent insights
    """
    try:
        # Get agent URL either from request or from config
        url = agent_url
        if not url:
            try:
                from app.core.config import settings
                agent_config = settings.get_agent_config(agent_type)
                url = agent_config["url"]
                timeout = agent_config.get("timeout", 30)
            except ValueError:
                return {"error": f"No URL configured for agent type {agent_type}"}
        else:
            timeout = 30  # Default timeout

        # Prepare context for the agent
        agent_context = context or {}
        agent_context.update({
            "original_prompt": prompt,
            "generated_sql": sql,
            "agent_type": agent_type
        })

        # Create A2A messages
        messages = [
            a2a.create_message(
                content=f"Please analyze this SQL query for a healthcare question:\n\nQuestion: {prompt}\n\nSQL: {sql}",
                role="user",
                metadata=agent_context
            )
        ]

        # Create A2A request
        request = a2a.create_request(messages)

        # Send request using A2A protocol
        response = await a2a.send_request(url, request, timeout)

        return response

    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="Get_Available_Agents",
    description="Get list of available agent types"
)
def get_available_agents() -> Dict[str, Dict[str, Any]]:
    """Get list of available agent types

    Returns:
        Dictionary of available agents and their configurations
    """
    try:
        from app.core.config import settings
        return settings.config.get("agents", {})
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="Agent_Health_Check",
    description="Check if an agent is available and responding"
)
async def agent_health_check(agent_type: str) -> Dict[str, Any]:
    """Check if an agent is available and responding

    Args:
        agent_type: Type of agent to check

    Returns:
        Health check result
    """
    try:
        from app.core.config import settings

        try:
            agent_config = settings.get_agent_config(agent_type)
            url = agent_config["url"]
            health_url = f"{url}/health"

            async with httpx.AsyncClient() as client:
                response = await client.get(health_url, timeout=5)
                response.raise_for_status()
                return {
                    "status": "available",
                    "details": response.json()
                }
        except ValueError:
            return {"status": "unavailable", "details": f"No configuration for agent type {agent_type}"}
    except Exception as e:
        return {"status": "error", "details": str(e)}


if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")
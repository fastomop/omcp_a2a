import httpx
from typing import Dict, Any, List, Optional


class A2AProtocol:
    """A2A protocol implementation for agent-to-agent communication"""

    @staticmethod
    def create_message(content: str, role: str = "user", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a message following A2A protocol"""
        return {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }

    @staticmethod
    def create_request(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create an A2A request"""
        request = {
            "messages": messages
        }

        if tools:
            request["tools"] = tools

        return request

    @staticmethod
    async def send_request(url: str, request_data: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """Send a request to another agent using A2A protocol"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=request_data,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
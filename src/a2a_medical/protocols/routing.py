"""
Message routing protocols for medical A2A communication.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class RoutingStrategy(Enum):
    """Routing strategies for message delivery."""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"


@dataclass
class Route:
    """Represents a message route."""
    
    route_id: str
    source: str
    destination: str
    strategy: RoutingStrategy
    priority: int = 1
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingTable:
    """Routing table for message routing."""
    
    routes: Dict[str, Route] = field(default_factory=dict)
    default_strategy: RoutingStrategy = RoutingStrategy.DIRECT
    
    def add_route(self, route: Route) -> None:
        """Add a route to the routing table."""
        self.routes[route.route_id] = route
    
    def remove_route(self, route_id: str) -> None:
        """Remove a route from the routing table."""
        if route_id in self.routes:
            del self.routes[route_id]
    
    def get_route(self, route_id: str) -> Optional[Route]:
        """Get a route by ID."""
        return self.routes.get(route_id)
    
    def get_routes_for_destination(self, destination: str) -> List[Route]:
        """Get all routes for a specific destination."""
        return [route for route in self.routes.values() 
                if route.destination == destination and route.is_active]


class MessageRouter(ABC):
    """Base class for message routing in medical A2A systems."""
    
    def __init__(self, router_id: str):
        self.router_id = router_id
        self.routing_table = RoutingTable()
        self.connected_agents: Set[str] = set()
        self.routing_stats: Dict[str, int] = {
            "messages_routed": 0,
            "routing_errors": 0,
            "messages_delivered": 0
        }
    
    @abstractmethod
    async def route_message(self, message: Any, source: str, destinations: List[str]) -> Dict[str, Any]:
        """Route a message to one or more destinations."""
        pass
    
    @abstractmethod
    async def add_route(self, route: Route) -> bool:
        """Add a new route to the routing table."""
        pass
    
    @abstractmethod
    async def remove_route(self, route_id: str) -> bool:
        """Remove a route from the routing table."""
        pass
    
    def register_agent(self, agent_id: str) -> None:
        """Register an agent with the router."""
        self.connected_agents.add(agent_id)
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the router."""
        self.connected_agents.discard(agent_id)
    
    def get_connected_agents(self) -> Set[str]:
        """Get all connected agents."""
        return self.connected_agents.copy()
    
    def get_routing_stats(self) -> Dict[str, int]:
        """Get routing statistics."""
        return self.routing_stats.copy()
    
    def update_stats(self, stat_name: str, increment: int = 1) -> None:
        """Update routing statistics."""
        if stat_name in self.routing_stats:
            self.routing_stats[stat_name] += increment


class SimpleMessageRouter(MessageRouter):
    """Simple implementation of message router."""
    
    async def route_message(self, message: Any, source: str, destinations: List[str]) -> Dict[str, Any]:
        """Route a message using simple direct routing."""
        results = {
            "routed": [],
            "failed": [],
            "total_destinations": len(destinations)
        }
        
        for destination in destinations:
            if destination in self.connected_agents:
                # Simulate successful routing
                results["routed"].append(destination)
                self.update_stats("messages_delivered")
            else:
                results["failed"].append(destination)
                self.update_stats("routing_errors")
        
        self.update_stats("messages_routed")
        return results
    
    async def add_route(self, route: Route) -> bool:
        """Add a new route to the routing table."""
        self.routing_table.add_route(route)
        return True
    
    async def remove_route(self, route_id: str) -> bool:
        """Remove a route from the routing table."""
        if route_id in self.routing_table.routes:
            self.routing_table.remove_route(route_id)
            return True
        return False


class LoadBalancedRouter(MessageRouter):
    """Load-balanced message router implementation."""
    
    def __init__(self, router_id: str):
        super().__init__(router_id)
        self.agent_loads: Dict[str, int] = {}
        self.current_round_robin_index = 0
    
    async def route_message(self, message: Any, source: str, destinations: List[str]) -> Dict[str, Any]:
        """Route a message using load balancing."""
        if not destinations:
            return {"routed": [], "failed": [], "total_destinations": 0}
        
        # Simple round-robin load balancing
        selected_destination = destinations[self.current_round_robin_index % len(destinations)]
        self.current_round_robin_index += 1
        
        results = {
            "routed": [selected_destination] if selected_destination in self.connected_agents else [],
            "failed": [selected_destination] if selected_destination not in self.connected_agents else [],
            "total_destinations": 1
        }
        
        if results["routed"]:
            self.update_stats("messages_delivered")
        else:
            self.update_stats("routing_errors")
        
        self.update_stats("messages_routed")
        return results
    
    async def add_route(self, route: Route) -> bool:
        """Add a new route to the routing table."""
        self.routing_table.add_route(route)
        return True
    
    async def remove_route(self, route_id: str) -> bool:
        """Remove a route from the routing table."""
        if route_id in self.routing_table.routes:
            self.routing_table.remove_route(route_id)
            return True
        return False

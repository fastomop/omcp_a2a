"""
Base Medical Agent implementation for the A2A Medical Foundation Framework.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import uuid
import json

# A2A SDK imports - confirmed available
from a2a.types import (
    AgentCard, Message, SendMessageRequest, SendMessageResponse, 
    AgentCapabilities, AgentSkill, AgentExtension,
    SendMessageSuccessResponse, TextPart, Role, Part
)
from a2a.client import A2AClient


# Type definitions
@dataclass
class ProcessedObservation:
    """Represents a processed observation from the environment."""
    data: Any
    timestamp: float
    source: str
    confidence: float = 1.0


@dataclass
class Action:
    """Represents an action to be executed by the agent."""
    action_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Represents the result of an executed action."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorldModel(ABC):
    """Abstract world model for representing agent's understanding of the environment.
    
    This replaces the simple beliefs system with a more sophisticated and flexible
    world modeling approach that can be specialized for different medical domains.
    """
    
    def __init__(self):
        self.model_id = str(uuid.uuid4())
        self.created_at = None
        self.last_updated = None
    
    @abstractmethod
    def update(self, observation: ProcessedObservation) -> None:
        """Update the world model with new observations."""
        pass
    
    @abstractmethod
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Query the world model for information."""
        pass
    
    @abstractmethod
    def predict(self, scenario: Dict[str, Any]) -> Any:
        """Make predictions based on the current world model."""
        pass
    
    @abstractmethod
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current world model state."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the world model to initial state."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get metadata about the world model."""
        return {
            "model_id": self.model_id,
            "model_type": self.__class__.__name__,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }


class MentalState:
    """Represents the mental state of a medical agent.
    
    Now uses a world_model instead of simple beliefs, along with goals and emotions.
    """
    
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.goals: List[str] = []
        self.emotions: Dict[str, float] = {}
        self.memory: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
    
    def update_world_model(self, observation: ProcessedObservation) -> None:
        """Update the world model with new observations."""
        self.world_model.update(observation)
    
    def query_world_model(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Query the world model for information."""
        return self.world_model.query(query, context)
    
    def predict_outcome(self, scenario: Dict[str, Any]) -> Any:
        """Predict outcomes using the world model."""
        return self.world_model.predict(scenario)
    
    def add_goal(self, goal: str) -> None:
        """Add a goal to the mental state."""
        if goal not in self.goals:
            self.goals.append(goal)
    
    def remove_goal(self, goal: str) -> None:
        """Remove a goal from the mental state."""
        if goal in self.goals:
            self.goals.remove(goal)
    
    def set_emotion(self, emotion: str, intensity: float) -> None:
        """Set an emotion intensity in the mental state."""
        self.emotions[emotion] = max(0.0, min(1.0, intensity))
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a complete summary of the mental state."""
        return {
            "world_model": self.world_model.get_state_summary(),
            "goals": self.goals.copy(),
            "emotions": self.emotions.copy(),
            "memory_keys": list(self.memory.keys()),
            "context_keys": list(self.context.keys())
        }


class MedicalAgent(ABC):
    """Abstract base class for all medical foundation agents.
    
    Implements the core agent architecture with:
    - Perception (P): Process incoming observations
    - Cognition (C): Learning and reasoning using world model
    - Action Execution (E): Execute internal/external actions
    
    This is now purely abstract - implementations must provide their own
    world model and cognitive strategies.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        world_model: WorldModel
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.mental_state = MentalState(world_model)
        self.discovery_service = None
        self._clients: Dict[str, A2AClient] = {}
        
    @abstractmethod
    async def perceive(self, observation: Any) -> ProcessedObservation:
        """Perception module: o_t = P(s_t, M_{t-1})
        
        Must be implemented by concrete agents to define how they
        process and interpret incoming observations.
        """
        pass
        
    @abstractmethod
    async def learn(self, state: MentalState, observation: ProcessedObservation) -> MentalState:
        """Learning component: M_t = L(M_{t-1}, a_{t-1}, o_t)
        
        Must be implemented by concrete agents to define how they
        update their world model and mental state.
        """
        pass
        
    @abstractmethod
    async def reason(self, state: MentalState) -> Action:
        """Reasoning component: a_t = R(M_t)
        
        Must be implemented by concrete agents to define their
        decision-making and action selection strategies.
        """
        pass
        
    @abstractmethod
    async def execute(self, action: Action) -> ActionResult:
        """Action execution: E(a_t) = a'_t
        
        Must be implemented by concrete agents to define how they
        execute actions in their specific domain.
        """
        pass
        
    async def process_request(self, request: Message) -> SendMessageResponse:
        """Main request processing pipeline.
        
        This orchestrates the PCE cycle but delegates the actual
        implementation to the abstract methods.
        """
        # Perception
        observation = await self.perceive(request)
        
        # Cognition
        self.mental_state = await self.learn(self.mental_state, observation)
        action = await self.reason(self.mental_state)
        
        # Execution
        result = await self.execute(action)
        
        return self._build_response(result)
    
    def _build_response(self, result: ActionResult) -> SendMessageResponse:
        """Build a response from an action result."""
        # Create a text part with the response data
        response_text = json.dumps({
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "metadata": result.metadata
        })
        
        # Create a Part using the discriminated union approach
        text_part = Part.model_validate({
            'kind': 'text', 
            'text': response_text
        })
        
        # Create a message with proper structure
        response_message = Message(
            messageId=str(uuid.uuid4()),
            parts=[text_part],
            role=Role.agent
        )
        
        # Create a success response using the proper A2A SDK structure
        success_response = SendMessageSuccessResponse(
            result=response_message
        )
        
        return SendMessageResponse(root=success_response)
    
    @abstractmethod
    def build_agent_card(self) -> AgentCard:
        """Build A2A agent card with medical extensions.
        
        Must be implemented by concrete agents to define their
        specific capabilities and configuration.
        """
        pass
    
    def add_client(self, client_id: str, client: A2AClient) -> None:
        """Add an A2A client for communication."""
        self._clients[client_id] = client
    
    def get_client(self, client_id: str) -> Optional[A2AClient]:
        """Get an A2A client by ID."""
        return self._clients.get(client_id)
    
    def remove_client(self, client_id: str) -> None:
        """Remove an A2A client."""
        if client_id in self._clients:
            del self._clients[client_id]
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities.copy(),
            "mental_state_summary": self.mental_state.get_state_summary(),
            "connected_clients": list(self._clients.keys())
        }


class CognitiveModule(ABC):
    """Abstract base class for cognitive modules.
    
    Cognitive modules are pluggable components that can be used
    to extend agent capabilities in specific domains.
    """
    
    def __init__(self, module_id: str):
        self.module_id = module_id
        self.active = True
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    async def process(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Process input data with given context.
        
        Must be implemented by concrete cognitive modules.
        """
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the cognitive module.
        
        Must be implemented by concrete cognitive modules.
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get the capabilities this module provides.
        
        Must be implemented by concrete cognitive modules.
        """
        pass
    
    def activate(self) -> None:
        """Activate this cognitive module."""
        self.active = True
    
    def deactivate(self) -> None:
        """Deactivate this cognitive module."""
        self.active = False
    
    def is_active(self) -> bool:
        """Check if this cognitive module is active."""
        return self.active
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get information about this cognitive module."""
        return {
            "module_id": self.module_id,
            "module_type": self.__class__.__name__,
            "active": self.active,
            "capabilities": self.get_capabilities(),
            "config": self.config.copy()
        }
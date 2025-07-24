"""
Base Medical Agent implementation for the A2A Medical Foundation Framework.

This module provides the core abstract base classes for medical agents that fully
comply with the A2A (Agent-to-Agent) protocol while adding medical-specific capabilities.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
import uuid
import json
import asyncio
from datetime import datetime

# A2A SDK imports - confirmed available
from a2a.types import (
    AgentCard, Message, SendMessageRequest, SendMessageResponse, 
    AgentCapabilities, AgentSkill, AgentExtension,
    SendMessageSuccessResponse, TextPart, Role, Part, Task, TaskStatus,
    JSONRPCErrorResponse, InternalError, MessageSendParams,
    TaskQueryParams, TaskIdParams, TaskPushNotificationConfig,
    DeleteTaskPushNotificationConfigParams,
    GetTaskPushNotificationConfigParams,
    ListTaskPushNotificationConfigParams,
    UnsupportedOperationError, TaskState
)
from a2a.client import A2AClient
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.context import ServerCallContext
from a2a.server.events.event_queue import Event
from a2a.utils.errors import ServerError
from collections.abc import AsyncGenerator


# Type definitions
@dataclass
class ProcessedObservation:
    """Represents a processed observation from the environment."""
    data: Any
    timestamp: float
    source: str
    confidence: float = 1.0
    observation_type: str = "general"
    processed_data: Optional[Dict[str, Any]] = None


@dataclass
class Action:
    """Represents an action to be executed by the agent."""
    action_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    target_agent_id: Optional[str] = None
    requires_a2a_communication: bool = False


@dataclass
class ActionResult:
    """Represents the result of an executed action."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    a2a_message: Optional[Message] = None


class WorldModel(ABC):
    """Abstract world model for representing agent's understanding of the environment.
    
    This replaces the simple beliefs system with a more sophisticated and flexible
    world modeling approach that can be specialized for different medical domains.
    Enhanced with A2A protocol awareness for multi-agent coordination.
    """
    
    def __init__(self):
        self.model_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.agent_registry: Dict[str, AgentCard] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
    
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
    
    def register_agent(self, agent_card: AgentCard) -> None:
        """Register another agent in the world model for future collaboration."""
        self.agent_registry[agent_card.name] = agent_card
        self.last_updated = datetime.now()
    
    def get_available_agents(self, capability_filter: Optional[List[str]] = None) -> List[AgentCard]:
        """Get list of available agents, optionally filtered by capabilities."""
        if not capability_filter:
            return list(self.agent_registry.values())
        
        filtered_agents = []
        for agent_card in self.agent_registry.values():
            if agent_card.capabilities and any(cap in str(agent_card.capabilities) for cap in capability_filter):
                filtered_agents.append(agent_card)
        return filtered_agents
    
    def record_collaboration(self, agent_id: str, interaction_type: str, result: Dict[str, Any]) -> None:
        """Record a collaboration event for learning and optimization."""
        self.collaboration_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "interaction_type": interaction_type,
            "result": result
        })
        self.last_updated = datetime.now()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get metadata about the world model."""
        return {
            "model_id": self.model_id,
            "model_type": self.__class__.__name__,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "registered_agents": len(self.agent_registry),
            "collaboration_events": len(self.collaboration_history)
        }


class MentalState:
    """Represents the mental state of a medical agent.
    
    Now uses a world_model instead of simple beliefs, along with goals and emotions.
    Enhanced with A2A protocol awareness for multi-agent coordination.
    """
    
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.goals: List[str] = []
        self.emotions: Dict[str, float] = {}
        self.memory: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.message_history: List[Message] = []
    
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
    
    def add_task(self, task: Task) -> None:
        """Add an active task to track."""
        if hasattr(task, 'id') and task.id:
            self.active_tasks[task.id] = task
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update the status of an active task."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].status = status
    
    def record_message(self, message: Message) -> None:
        """Record a message in the conversation history."""
        self.message_history.append(message)
        # Keep only last 100 messages to prevent memory bloat
        if len(self.message_history) > 100:
            self.message_history = self.message_history[-100:]
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a complete summary of the mental state."""
        return {
            "world_model": self.world_model.get_state_summary(),
            "goals": self.goals.copy(),
            "emotions": self.emotions.copy(),
            "memory_keys": list(self.memory.keys()),
            "context_keys": list(self.context.keys()),
            "active_tasks": len(self.active_tasks),
            "message_history_length": len(self.message_history)
        }


class MedicalAgent(RequestHandler, ABC):
    """Abstract base class for all medical agents with full A2A protocol compliance.
    
    Implements the core agent architecture with:
    - Perception (P): Process incoming observations and A2A messages
    - Cognition (C): Learning and reasoning using world model
    - Action Execution (E): Execute internal/external actions and A2A communications
    
    This class provides full A2A protocol compliance while maintaining medical-specific
    abstractions. Concrete implementations must provide their own world model and 
    cognitive strategies.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        world_model: WorldModel,
        agent_name: Optional[str] = None,
        agent_description: Optional[str] = None,
        agent_version: str = "1.0.0"
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.agent_name = agent_name or f"{agent_type}-{agent_id}"
        self.agent_description = agent_description or f"Medical {agent_type} agent"
        self.agent_version = agent_version
        self.mental_state = MentalState(world_model)
        self.discovery_service = None
        self._clients: Dict[str, A2AClient] = {}
        self._message_handlers: Dict[str, Callable] = {}
        self._is_active = True
        
        # A2A Protocol compliance
        self._agent_card: Optional[AgentCard] = None
        self._task_handlers: Dict[str, Callable] = {}
        
    # Core PCE Architecture (Abstract methods)
    
    @abstractmethod
    async def perceive(self, observation: Any) -> ProcessedObservation:
        """Perception module: o_t = P(s_t, M_{t-1})
        
        Must be implemented by concrete agents to define how they
        process and interpret incoming observations and A2A messages.
        """
        pass
        
    @abstractmethod
    async def learn(self, state: MentalState, observation: ProcessedObservation) -> MentalState:
        """Learning component: M_t = L(M_{t-1}, a_{t-1}, o_t)
        
        Must be implemented by concrete agents to define how they
        update their world model and mental state based on observations
        and A2A interactions.
        """
        pass
        
    @abstractmethod
    async def reason(self, state: MentalState) -> Action:
        """Reasoning component: a_t = R(M_t)
        
        Must be implemented by concrete agents to define their
        decision-making and action selection strategies, including
        when to initiate A2A communications.
        """
        pass
        
    @abstractmethod
    async def execute(self, action: Action) -> ActionResult:
        """Action execution: E(a_t) = a'_t
        
        Must be implemented by concrete agents to define how they
        execute actions, including A2A message sending and task management.
        """
        pass
    
    # A2A RequestHandler Implementation
    
    async def on_message_send(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> Task | Message:
        """Handles the 'message/send' method (non-streaming).
        
        This method integrates A2A message processing with the PCE cycle.
        """
        try:
            # Record the message
            self.mental_state.record_message(params.message)
            
            # Perceive the message as an observation
            observation = await self.perceive(params.message)
            
            # Learn from the message
            self.mental_state = await self.learn(self.mental_state, observation)
            
            # Reason about the response
            action = await self.reason(self.mental_state)
            
            # Execute the action
            result = await self.execute(action)
            
            # Build and return A2A response
            return self._build_message_response(result)
            
        except Exception as e:
            # Create error message
            from a2a.types import Part
            text_part = TextPart(text=f"Error processing message: {str(e)}")
            parts = [Part(root=text_part)]
            return Message(
                messageId=str(uuid.uuid4()),
                parts=parts,
                role=Role.agent
            )
    
    async def on_message_send_stream(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> AsyncGenerator[Event]:
        """Handles the 'message/stream' method (streaming)."""
        # Default implementation raises UnsupportedOperationError
        raise ServerError(error=UnsupportedOperationError())
        yield  # Required for generator
    
    async def on_get_task(
        self,
        params: TaskQueryParams,
        context: ServerCallContext | None = None,
    ) -> Task | None:
        """Handles the 'tasks/get' method."""
        # Default implementation - concrete agents can override
        task_id = params.id if hasattr(params, 'id') else None
        if task_id and task_id in self.mental_state.active_tasks:
            return self.mental_state.active_tasks[task_id]
        return None
    
    async def on_cancel_task(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ) -> Task | None:
        """Handles the 'tasks/cancel' method."""
        # Default implementation - concrete agents can override
        task_id = params.id if hasattr(params, 'id') else None
        if task_id and task_id in self.mental_state.active_tasks:
            task = self.mental_state.active_tasks[task_id]
            # Update task status to cancelled if possible
            if hasattr(task, 'status'):
                task.status.state = TaskState.canceled
            return task
        return None
    
    async def on_set_task_push_notification_config(
        self,
        params: TaskPushNotificationConfig,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        """Handles the 'tasks/pushNotificationConfig/set' method."""
        # Default implementation - concrete agents can override
        return params
    
    async def on_get_task_push_notification_config(
        self,
        params: TaskIdParams | GetTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        """Handles the 'tasks/pushNotificationConfig/get' method."""
        # Default implementation - concrete agents can override
        raise ServerError(error=UnsupportedOperationError())
    
    async def on_resubscribe_to_task(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ) -> AsyncGenerator[Event]:
        """Handles the 'tasks/resubscribe' method."""
        # Default implementation raises UnsupportedOperationError
        raise ServerError(error=UnsupportedOperationError())
        yield  # Required for generator
    
    async def on_list_task_push_notification_config(
        self,
        params: ListTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> list[TaskPushNotificationConfig]:
        """Handles the 'tasks/pushNotificationConfig/list' method."""
        # Default implementation - concrete agents can override
        return []
    
    async def on_delete_task_push_notification_config(
        self,
        params: DeleteTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> None:
        """Handles the 'tasks/pushNotificationConfig/delete' method."""
        # Default implementation - concrete agents can override
        pass
    
    def _build_message_response(self, result: ActionResult) -> Message:
        """Build an A2A message response from an action result."""
        from a2a.types import Part
        parts = []
        if result.success and result.data:
            if isinstance(result.data, str):
                text_part = TextPart(text=result.data)
                parts.append(Part(root=text_part))
            elif isinstance(result.data, dict):
                text_part = TextPart(text=json.dumps(result.data, default=str))
                parts.append(Part(root=text_part))
            else:
                text_part = TextPart(text=str(result.data))
                parts.append(Part(root=text_part))
        elif result.error:
            text_part = TextPart(text=f"Error: {result.error}")
            parts.append(Part(root=text_part))
        else:
            text_part = TextPart(text="Task completed successfully")
            parts.append(Part(root=text_part))
        
        return Message(
            messageId=str(uuid.uuid4()),
            parts=parts,
            role=Role.agent
        )
    
    @abstractmethod
    def build_agent_card(self) -> AgentCard:
        """Build the A2A agent card for this medical agent.
        
        Must be implemented by concrete agents to define their
        A2A capabilities, endpoints, and metadata.
        """
        pass
    
    # A2A Client Management
    
    def add_client(self, client_id: str, client: A2AClient) -> None:
        """Add an A2A client for communicating with other agents."""
        self._clients[client_id] = client
    
    def get_client(self, client_id: str) -> Optional[A2AClient]:
        """Get an A2A client by ID."""
        return self._clients.get(client_id)
    
    def remove_client(self, client_id: str) -> None:
        """Remove an A2A client."""
        if client_id in self._clients:
            del self._clients[client_id]
    
    # Agent Information and Management
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information including A2A details."""
        agent_card = self.build_agent_card()
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "agent_description": self.agent_description,
            "capabilities": self.capabilities,
            "mental_state": self.mental_state.get_state_summary(),
            "is_active": self._is_active,
            "connected_clients": list(self._clients.keys()),
            "agent_card": {
                "name": agent_card.name,
                "description": agent_card.description,
                "version": agent_card.version,
                "url": agent_card.url if hasattr(agent_card, 'url') else None
            }
        }
    
    # Message and Task Handlers
    
    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register a handler for specific message types."""
        self._message_handlers[message_type] = handler
    
    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler for specific task types."""
        self._task_handlers[task_type] = handler
    
    async def send_message_to_agent(self, target_agent_id: str, message: str, context: Optional[Dict[str, Any]] = None) -> Optional[SendMessageResponse]:
        """Send a message to another agent via A2A protocol."""
        client = self.get_client(target_agent_id)
        if not client:
            return None
        
        # Create proper A2A message parts
        from a2a.types import Part
        text_part = TextPart(text=message)
        parts = [Part(root=text_part)]
        
        # Ensure a unique message ID is always generated
        message_id = str(uuid.uuid4())

        request_message = Message(
            messageId=message_id,
            parts=parts,
            role=Role.user
        )
        
        try:
            # Create proper SendMessageRequest
            from a2a.types import MessageSendParams, SendMessageSuccessResponse
            params = MessageSendParams(message=request_message)
            send_request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=params
            )
            
            response = await client.send_message(send_request)
            self.mental_state.record_message(request_message)
            
            # Handle response safely - check if it's a success response
            if hasattr(response, 'root') and isinstance(response.root, SendMessageSuccessResponse):
                # This is a success response, record the returned message if it's a Message
                result_data = response.root.result
                if isinstance(result_data, Message):
                    self.mental_state.record_message(result_data)
            
            return response
        except Exception as e:
            error = InternalError(message=str(e))
            error_response = JSONRPCErrorResponse(error=error)
            return SendMessageResponse(root=error_response)
    
    def activate(self) -> None:
        """Activate the agent for A2A communications."""
        self._is_active = True
    
    def deactivate(self) -> None:
        """Deactivate the agent."""
        self._is_active = False
    
    def is_active(self) -> bool:
        """Check if the agent is active."""
        return self._is_active


class CognitiveModule(ABC):
    """Abstract base class for pluggable cognitive modules.
    
    Enhanced with A2A protocol awareness for distributed cognitive processing.
    """
    
    def __init__(self, module_id: str):
        self.module_id = module_id
        self.is_active_flag = False
        self.config: Dict[str, Any] = {}
        self.dependencies: List[str] = []
        self.a2a_enabled = False
    
    @abstractmethod
    async def process(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Process input data and return results.
        
        Can now coordinate with other cognitive modules via A2A if enabled.
        """
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the cognitive module with parameters.
        
        Can include A2A endpoint configurations for distributed processing.
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this module provides."""
        pass
    
    def enable_a2a(self, client: A2AClient) -> None:
        """Enable A2A communications for this cognitive module."""
        self.a2a_client = client
        self.a2a_enabled = True
    
    def activate(self) -> None:
        """Activate the cognitive module."""
        self.is_active_flag = True
    
    def deactivate(self) -> None:
        """Deactivate the cognitive module."""
        self.is_active_flag = False
    
    def is_active(self) -> bool:
        """Check if the module is active."""
        return self.is_active_flag
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get module information including A2A status."""
        return {
            "module_id": self.module_id,
            "module_type": self.__class__.__name__,
            "is_active": self.is_active_flag,
            "capabilities": self.get_capabilities(),
            "dependencies": self.dependencies,
            "a2a_enabled": self.a2a_enabled,
            "config_keys": list(self.config.keys())
        }
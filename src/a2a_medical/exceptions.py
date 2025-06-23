"""
Custom exceptions for the Medical A2A Foundation Framework.
"""

from typing import Any, Optional


class A2AMedicalError(Exception):
    """Base exception for all Medical A2A framework errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(A2AMedicalError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None) -> None:
        super().__init__(message, {"field": field, "value": value})
        self.field = field
        self.value = value


class ComplianceError(A2AMedicalError):
    """Raised when regulatory compliance requirements are not met."""
    
    def __init__(self, message: str, regulation: Optional[str] = None, requirement: Optional[str] = None) -> None:
        super().__init__(message, {"regulation": regulation, "requirement": requirement})
        self.regulation = regulation
        self.requirement = requirement


class SecurityError(A2AMedicalError):
    """Raised when security requirements are violated."""
    
    def __init__(self, message: str, security_level: Optional[str] = None, violation_type: Optional[str] = None) -> None:
        super().__init__(message, {"security_level": security_level, "violation_type": violation_type})
        self.security_level = security_level
        self.violation_type = violation_type


class ProtocolError(A2AMedicalError):
    """Raised when A2A protocol communication fails."""
    
    def __init__(self, message: str, protocol: Optional[str] = None, endpoint: Optional[str] = None) -> None:
        super().__init__(message, {"protocol": protocol, "endpoint": endpoint})
        self.protocol = protocol
        self.endpoint = endpoint


class AgentError(A2AMedicalError):
    """Raised when agent-related operations fail."""
    
    def __init__(self, message: str, agent_id: Optional[str] = None, operation: Optional[str] = None) -> None:
        super().__init__(message, {"agent_id": agent_id, "operation": operation})
        self.agent_id = agent_id
        self.operation = operation


class MentalStateError(A2AMedicalError):
    """Raised when mental state operations fail."""
    
    def __init__(self, message: str, state_type: Optional[str] = None, transition: Optional[str] = None) -> None:
        super().__init__(message, {"state_type": state_type, "transition": transition})
        self.state_type = state_type
        self.transition = transition


class DiscoveryError(A2AMedicalError):
    """Raised when agent discovery operations fail."""
    
    def __init__(self, message: str, discovery_method: Optional[str] = None, target: Optional[str] = None) -> None:
        super().__init__(message, {"discovery_method": discovery_method, "target": target})
        self.discovery_method = discovery_method
        self.target = target


class RoutingError(A2AMedicalError):
    """Raised when message routing fails."""
    
    def __init__(self, message: str, route: Optional[str] = None, destination: Optional[str] = None) -> None:
        super().__init__(message, {"route": route, "destination": destination})
        self.route = route
        self.destination = destination

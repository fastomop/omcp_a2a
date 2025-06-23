"""
Medical A2A protocols and extensions.

This module provides abstract protocol definitions for medical A2A systems.
Concrete implementations should inherit from these abstractions.
"""

from typing import Protocol, List, Optional, Any
from abc import ABC, abstractmethod

# A2A SDK imports - confirmed available
from a2a.types import AgentCard, Message
from a2a.server.apps import A2AStarletteApplication
from a2a.client import A2AClient


class MedicalCapability:
    """Represents a medical capability of an agent."""
    
    def __init__(self, name: str, description: str, compliance_level: str = "HIPAA"):
        self.name = name
        self.description = description
        self.compliance_level = compliance_level


class EmergencyRequest:
    """Represents an emergency medical request."""
    
    def __init__(self, priority: int, description: str, patient_context: Optional[Any] = None):
        self.priority = priority
        self.description = description
        self.patient_context = patient_context


class EmergencyResponse:
    """Represents a response to an emergency medical request."""
    
    def __init__(self, status: str, recommendations: List[str], escalation_needed: bool = False):
        self.status = status
        self.recommendations = recommendations
        self.escalation_needed = escalation_needed


class ComplianceResult:
    """Result of a compliance validation."""
    
    def __init__(self, is_compliant: bool, violations: Optional[List[str]] = None, warnings: Optional[List[str]] = None):
        self.is_compliant = is_compliant
        self.violations = violations or []
        self.warnings = warnings or []


class CredentialVerificationResult:
    """Result of credential verification."""
    
    def __init__(self, is_verified: bool, credentials: Optional[List[str]] = None, expires_at: Optional[Any] = None):
        self.is_verified = is_verified
        self.credentials = credentials or []
        self.expires_at = expires_at


class MedicalAgentProtocol(Protocol):
    """Protocol interface for medical agents in the A2A ecosystem."""
    
    async def validate_medical_compliance(self, request: Any) -> ComplianceResult:
        """Validate request meets medical compliance requirements."""
        ...
        
    async def get_medical_capabilities(self) -> List[MedicalCapability]:
        """Return medical-specific capabilities."""
        ...
        
    async def handle_emergency_request(self, request: EmergencyRequest) -> EmergencyResponse:
        """Handle high-priority medical requests."""
        ...


class MedicalDiscoveryProtocol(Protocol):
    """Protocol interface for discovering medical agents."""
    
    async def find_by_medical_capability(
        self,
        capability: str,
        compliance_level: str = "HIPAA"
    ) -> List[AgentCard]:
        """Find agents with specific medical capabilities."""
        ...
        
    async def verify_medical_credentials(
        self,
        agent_id: str
    ) -> CredentialVerificationResult:
        """Verify medical agent credentials."""
        ...


class MedicalA2AProtocol(ABC):
    """Abstract base class for medical A2A protocol extensions.
    
    Provides the foundational structure for implementing medical-specific
    A2A protocols. Concrete implementations should inherit from this class.
    """
    
    def __init__(self, protocol_id: str):
        self.protocol_id = protocol_id
        self.medical_capabilities: List[MedicalCapability] = []
        self.compliance_validators: List[Any] = []
    
    @abstractmethod
    async def validate_medical_request(self, request: Any) -> ComplianceResult:
        """Validate that a request meets medical compliance requirements.
        
        Must be implemented by concrete protocol classes.
        """
        pass
    
    @abstractmethod
    async def handle_medical_emergency(self, emergency: EmergencyRequest) -> EmergencyResponse:
        """Handle medical emergency requests with appropriate priority.
        
        Must be implemented by concrete protocol classes.
        """
        pass
    
    @abstractmethod
    async def discover_medical_agents(self, capability: str) -> List[AgentCard]:
        """Discover agents with specific medical capabilities.
        
        Must be implemented by concrete protocol classes.
        """
        pass
    
    def add_medical_capability(self, capability: MedicalCapability) -> None:
        """Add a medical capability to this protocol."""
        self.medical_capabilities.append(capability)
    
    def remove_medical_capability(self, capability_name: str) -> None:
        """Remove a medical capability from this protocol."""
        self.medical_capabilities = [
            cap for cap in self.medical_capabilities 
            if cap.name != capability_name
        ]
    
    def get_medical_capabilities(self) -> List[MedicalCapability]:
        """Get all medical capabilities of this protocol."""
        return self.medical_capabilities.copy()
    
    def add_compliance_validator(self, validator: Any) -> None:
        """Add a compliance validator to this protocol."""
        self.compliance_validators.append(validator)
    
    def get_protocol_info(self) -> dict:
        """Get information about this protocol."""
        return {
            "protocol_id": self.protocol_id,
            "protocol_type": self.__class__.__name__,
            "capabilities_count": len(self.medical_capabilities),
            "validators_count": len(self.compliance_validators)
        }


class ComplianceProtocol(ABC):
    """Abstract base class for compliance protocols.
    
    Provides foundation for implementing various healthcare compliance
    standards (HIPAA, GDPR, HITECH, etc.).
    """
    
    def __init__(self, standard_name: str, version: str = "1.0"):
        self.standard_name = standard_name
        self.version = version
        self.rules: List[Any] = []
        self.validators: List[Any] = []
    
    @abstractmethod
    async def validate_compliance(self, request: Any) -> ComplianceResult:
        """Validate compliance against this standard.
        
        Must be implemented by concrete compliance protocols.
        """
        pass
    
    @abstractmethod
    def get_compliance_requirements(self) -> List[str]:
        """Get list of compliance requirements for this standard.
        
        Must be implemented by concrete compliance protocols.
        """
        pass
    
    @abstractmethod
    def configure_validation_rules(self, config: dict) -> None:
        """Configure validation rules for this compliance standard.
        
        Must be implemented by concrete compliance protocols.
        """
        pass
    
    def get_standard_info(self) -> dict:
        """Get information about this compliance standard."""
        return {
            "standard_name": self.standard_name,
            "version": self.version,
            "rules_count": len(self.rules),
            "validators_count": len(self.validators)
        }


class EmergencyProtocol(ABC):
    """Abstract base class for emergency handling protocols.
    
    Provides foundation for implementing emergency response and
    escalation procedures in medical systems.
    """
    
    def __init__(self, protocol_name: str):
        self.protocol_name = protocol_name
        self.priority_levels: List[int] = []
        self.escalation_rules: dict = {}
    
    @abstractmethod
    async def assess_emergency_level(self, request: EmergencyRequest) -> int:
        """Assess the emergency level of a request.
        
        Must be implemented by concrete emergency protocols.
        """
        pass
    
    @abstractmethod
    async def handle_emergency(self, request: EmergencyRequest) -> EmergencyResponse:
        """Handle an emergency request.
        
        Must be implemented by concrete emergency protocols.
        """
        pass
    
    @abstractmethod
    def configure_escalation_rules(self, rules: dict) -> None:
        """Configure escalation rules for emergencies.
        
        Must be implemented by concrete emergency protocols.
        """
        pass
    
    def get_protocol_info(self) -> dict:
        """Get information about this emergency protocol."""
        return {
            "protocol_name": self.protocol_name,
            "priority_levels": self.priority_levels.copy(),
            "escalation_rules_count": len(self.escalation_rules)
        }
"""
Medical Application-to-Application (A2A) Foundation Framework.

A comprehensive framework for building secure, compliant, and efficient
medical data integration systems using agent-based architectures.

This package provides abstract foundations for building medical A2A systems.
Concrete implementations should inherit from these abstract base classes.
"""

__version__ = "0.1.0"
__author__ = "Medical A2A Foundation Team"
__email__ = "foundation@medical-a2a.org"

# Core abstract base classes
from .base.agent import MedicalAgent, WorldModel, MentalState, CognitiveModule

# Model imports (concrete Pydantic models)
from .models.medical import MedicalRecord, Patient, Provider
from .models.messages import A2AMessage, MessageType
from .models.tasks import MedicalTask, TaskStatus
from .models.omop import OMOPConcept, OMOPVocabulary

# Protocol abstract base classes
from .protocols.medical_a2a import MedicalA2AProtocol, ComplianceProtocol, EmergencyProtocol
from .protocols.discovery import AgentDiscovery, MedicalAgentDiscovery, ComplianceChecker, AgentRegistry
from .protocols.routing import MessageRouter

# Validator abstract base classes
from .validators.query import QueryValidator
from .validators.safety import SafetyValidator, MedicalComplianceValidator, DrugInteractionValidator, DosageValidator, EmergencyValidator
from .validators.compliance import ComplianceValidator

# Utility abstract base classes
from .utils.logging import MedicalLogger, ComplianceLogger, PHIDetector, LogSanitizer, SecureLogger
from .utils.metrics import MetricsCollector, MedicalMetricsCollector, PerformanceMonitor, ComplianceMonitor, SecurityMetricsCollector
from .utils.crypto import CryptoManager

# Exception imports
from .exceptions import (
    A2AMedicalError,
    ValidationError,
    ComplianceError,
    SecurityError,
    ProtocolError,
)

__all__ = [
    # Core abstract base classes
    "MedicalAgent",
    "WorldModel",
    "MentalState", 
    "CognitiveModule",
    
    # Models (concrete)
    "MedicalRecord",
    "Patient",
    "Provider",
    "A2AMessage",
    "MessageType",
    "MedicalTask",
    "TaskStatus",
    "OMOPConcept",
    "OMOPVocabulary",
    
    # Protocol abstract base classes
    "MedicalA2AProtocol",
    "ComplianceProtocol",
    "EmergencyProtocol",
    "AgentDiscovery",
    "MedicalAgentDiscovery",
    "ComplianceChecker",
    "AgentRegistry",
    "MessageRouter",
    
    # Validator abstract base classes
    "QueryValidator",
    "SafetyValidator",
    "MedicalComplianceValidator",
    "DrugInteractionValidator",
    "DosageValidator",
    "EmergencyValidator",
    "ComplianceValidator",
    
    # Utility abstract base classes
    "MedicalLogger",
    "ComplianceLogger",
    "PHIDetector",
    "LogSanitizer",
    "SecureLogger",
    "MetricsCollector",
    "MedicalMetricsCollector",
    "PerformanceMonitor",
    "ComplianceMonitor",
    "SecurityMetricsCollector",
    "CryptoManager",
    
    # Exceptions
    "A2AMedicalError",
    "ValidationError",
    "ComplianceError",
    "SecurityError",
    "ProtocolError",
]

# Framework metadata
FRAMEWORK_INFO = {
    "name": "A2A Medical Foundation",
    "version": __version__,
    "description": "Abstract foundation framework for medical A2A systems",
    "architecture": "Agent-based with PCE (Perception-Cognition-Execution) paradigm",
    "compliance_standards": ["HIPAA", "GDPR", "HITECH", "FDA 21 CFR Part 11"],
    "key_features": [
        "Abstract agent foundations",
        "World model-based cognition",
        "Medical compliance protocols",
        "Safety validation frameworks",
        "Security and encryption abstractions",
        "Performance monitoring foundations",
        "Discovery and routing protocols"
    ]
}

"""
Medical logging utilities for A2A medical systems.

This module provides abstract logging frameworks that can be specialized
for different medical domains and logging requirements.
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
import json


@dataclass
class LogEntry:
    """Represents a medical log entry."""
    timestamp: datetime
    level: str
    message: str
    agent_id: Optional[str] = None
    patient_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    phi_detected: bool = False


@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_id: str
    event_type: str
    timestamp: datetime
    actor: str
    resource: str
    action: str
    outcome: str
    metadata: Optional[Dict[str, Any]] = None


class MedicalLogger(ABC):
    """Abstract base class for medical logging systems.
    
    Provides foundational structure for implementing medical-compliant
    logging with PHI protection and audit capabilities.
    """
    
    def __init__(self, logger_id: str, phi_protection: bool = True):
        self.logger_id = logger_id
        self.phi_protection = phi_protection
        self.audit_enabled = True
        self.phi_detector = None
        self.sanitizer = None
    
    @abstractmethod
    def log_medical_event(self, 
                         level: str, 
                         message: str, 
                         agent_id: Optional[str] = None,
                         patient_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a medical event with appropriate protection.
        
        Must be implemented by concrete medical loggers.
        """
        pass
    
    @abstractmethod
    def log_audit_event(self, audit_event: AuditEvent) -> None:
        """Log an audit event for compliance tracking.
        
        Must be implemented by concrete medical loggers.
        """
        pass
    
    @abstractmethod
    def configure_phi_protection(self, config: Dict[str, Any]) -> None:
        """Configure PHI (Protected Health Information) protection.
        
        Must be implemented by concrete medical loggers.
        """
        pass
    
    @abstractmethod
    def search_logs(self, 
                   criteria: Dict[str, Any], 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[LogEntry]:
        """Search logs based on criteria.
        
        Must be implemented by concrete medical loggers.
        """
        pass
    
    @abstractmethod
    def export_audit_trail(self, 
                          start_time: datetime, 
                          end_time: datetime,
                          format: str = "json") -> str:
        """Export audit trail for compliance reporting.
        
        Must be implemented by concrete medical loggers.
        """
        pass
    
    def enable_phi_protection(self) -> None:
        """Enable PHI protection for this logger."""
        self.phi_protection = True
    
    def disable_phi_protection(self) -> None:
        """Disable PHI protection for this logger."""
        self.phi_protection = False
    
    def enable_audit(self) -> None:
        """Enable audit logging."""
        self.audit_enabled = True
    
    def disable_audit(self) -> None:
        """Disable audit logging."""
        self.audit_enabled = False
    
    def get_logger_info(self) -> Dict[str, Any]:
        """Get information about this logger."""
        return {
            "logger_id": self.logger_id,
            "logger_type": self.__class__.__name__,
            "phi_protection": self.phi_protection,
            "audit_enabled": self.audit_enabled
        }


class ComplianceLogger(ABC):
    """Abstract base class for compliance-focused logging.
    
    Provides foundation for implementing logging systems that
    comply with specific healthcare regulations.
    """
    
    def __init__(self, logger_id: str, compliance_standards: List[str]):
        self.logger_id = logger_id
        self.compliance_standards = compliance_standards
        self.retention_policies: Dict[str, int] = {}
        self.access_controls: Dict[str, List[str]] = {}
    
    @abstractmethod
    def log_compliance_event(self, 
                           standard: str, 
                           event_type: str, 
                           details: Dict[str, Any]) -> None:
        """Log a compliance-related event.
        
        Must be implemented by concrete compliance loggers.
        """
        pass
    
    @abstractmethod
    def validate_retention_policy(self, log_type: str) -> bool:
        """Validate that logs comply with retention policies.
        
        Must be implemented by concrete compliance loggers.
        """
        pass
    
    @abstractmethod
    def configure_retention_policy(self, log_type: str, retention_days: int) -> None:
        """Configure log retention policy.
        
        Must be implemented by concrete compliance loggers.
        """
        pass
    
    @abstractmethod
    def generate_compliance_report(self, 
                                 standard: str, 
                                 start_time: datetime, 
                                 end_time: datetime) -> Dict[str, Any]:
        """Generate compliance report for a specific standard.
        
        Must be implemented by concrete compliance loggers.
        """
        pass
    
    def add_compliance_standard(self, standard: str) -> None:
        """Add a compliance standard to this logger."""
        if standard not in self.compliance_standards:
            self.compliance_standards.append(standard)
    
    def get_supported_standards(self) -> List[str]:
        """Get list of supported compliance standards."""
        return self.compliance_standards.copy()


class PHIDetector(ABC):
    """Abstract base class for PHI detection systems.
    
    Provides foundation for implementing systems that can detect
    and handle Protected Health Information in logs.
    """
    
    def __init__(self, detector_id: str):
        self.detector_id = detector_id
        self.detection_patterns: List[str] = []
        self.sensitivity_level: str = "high"
    
    @abstractmethod
    def detect_phi(self, text: str) -> List[Dict[str, Any]]:
        """Detect PHI in the given text.
        
        Must be implemented by concrete PHI detectors.
        """
        pass
    
    @abstractmethod
    def configure_detection_patterns(self, patterns: List[str]) -> None:
        """Configure PHI detection patterns.
        
        Must be implemented by concrete PHI detectors.
        """
        pass
    
    @abstractmethod
    def set_sensitivity_level(self, level: str) -> None:
        """Set the sensitivity level for PHI detection.
        
        Must be implemented by concrete PHI detectors.
        """
        pass
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get PHI detection statistics."""
        return {
            "detector_id": self.detector_id,
            "patterns_count": len(self.detection_patterns),
            "sensitivity_level": self.sensitivity_level
        }


class LogSanitizer(ABC):
    """Abstract base class for log sanitization systems.
    
    Provides foundation for implementing systems that can sanitize
    logs by removing or masking sensitive information.
    """
    
    def __init__(self, sanitizer_id: str):
        self.sanitizer_id = sanitizer_id
        self.sanitization_rules: List[Dict[str, Any]] = []
        self.masking_strategy: str = "partial"
    
    @abstractmethod
    def sanitize_message(self, message: str, phi_locations: List[Dict[str, Any]]) -> str:
        """Sanitize a log message by removing or masking PHI.
        
        Must be implemented by concrete sanitizers.
        """
        pass
    
    @abstractmethod
    def configure_sanitization_rules(self, rules: List[Dict[str, Any]]) -> None:
        """Configure sanitization rules.
        
        Must be implemented by concrete sanitizers.
        """
        pass
    
    @abstractmethod
    def set_masking_strategy(self, strategy: str) -> None:
        """Set the masking strategy for sanitization.
        
        Must be implemented by concrete sanitizers.
        """
        pass
    
    def get_sanitizer_info(self) -> Dict[str, Any]:
        """Get information about this sanitizer."""
        return {
            "sanitizer_id": self.sanitizer_id,
            "rules_count": len(self.sanitization_rules),
            "masking_strategy": self.masking_strategy
        }


class SecureLogger(ABC):
    """Abstract base class for secure logging systems.
    
    Provides foundation for implementing logging systems with
    encryption, integrity protection, and secure storage.
    """
    
    def __init__(self, logger_id: str, encryption_enabled: bool = True):
        self.logger_id = logger_id
        self.encryption_enabled = encryption_enabled
        self.integrity_protection = True
        self.secure_storage = True
    
    @abstractmethod
    def log_secure_event(self, 
                        event: LogEntry, 
                        encryption_level: str = "aes256") -> None:
        """Log an event with security protection.
        
        Must be implemented by concrete secure loggers.
        """
        pass
    
    @abstractmethod
    def verify_log_integrity(self, log_id: str) -> bool:
        """Verify the integrity of a log entry.
        
        Must be implemented by concrete secure loggers.
        """
        pass
    
    @abstractmethod
    def configure_encryption(self, config: Dict[str, Any]) -> None:
        """Configure encryption settings.
        
        Must be implemented by concrete secure loggers.
        """
        pass
    
    @abstractmethod
    def rotate_encryption_keys(self) -> None:
        """Rotate encryption keys for security.
        
        Must be implemented by concrete secure loggers.
        """
        pass
    
    def enable_encryption(self) -> None:
        """Enable encryption for this logger."""
        self.encryption_enabled = True
    
    def disable_encryption(self) -> None:
        """Disable encryption for this logger."""
        self.encryption_enabled = False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security status of this logger."""
        return {
            "logger_id": self.logger_id,
            "encryption_enabled": self.encryption_enabled,
            "integrity_protection": self.integrity_protection,
            "secure_storage": self.secure_storage
        }
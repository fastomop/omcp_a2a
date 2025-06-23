"""
Compliance validation for medical A2A operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..exceptions import ComplianceError


class ComplianceStandard(Enum):
    """Compliance standards for medical systems."""
    HIPAA = "hipaa"
    GDPR = "gdpr"
    HITECH = "hitech"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"


class ComplianceLevel(Enum):
    """Compliance validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    AUDIT = "audit"


@dataclass
class ComplianceResult:
    """Result of a compliance validation."""
    
    is_compliant: bool
    standard: ComplianceStandard
    level: ComplianceLevel
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_violation(self, violation: str) -> None:
        """Add a compliance violation."""
        self.violations.append(violation)
        self.is_compliant = False
    
    def add_warning(self, warning: str) -> None:
        """Add a compliance warning."""
        self.warnings.append(warning)
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a compliance recommendation."""
        self.recommendations.append(recommendation)
    
    def has_violations(self) -> bool:
        """Check if there are compliance violations."""
        return len(self.violations) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are compliance warnings."""
        return len(self.warnings) > 0


class ComplianceValidator(ABC):
    """Base class for compliance validation in medical A2A systems."""
    
    def __init__(self, standard: ComplianceStandard, level: ComplianceLevel = ComplianceLevel.STANDARD):
        self.standard = standard
        self.level = level
        self.compliance_rules: Dict[str, Any] = {}
        self.validation_stats: Dict[str, int] = {
            "validations_performed": 0,
            "compliance_violations": 0,
            "compliance_warnings": 0
        }
    
    @abstractmethod
    async def validate_compliance(self, data: Any) -> ComplianceResult:
        """Validate compliance of data according to the standard."""
        pass
    
    @abstractmethod
    async def validate_data_privacy(self, data: Any) -> ComplianceResult:
        """Validate data privacy compliance."""
        pass
    
    @abstractmethod
    async def validate_security(self, data: Any) -> ComplianceResult:
        """Validate security compliance."""
        pass
    
    def add_compliance_rule(self, rule_name: str, rule: Any) -> None:
        """Add a compliance rule."""
        self.compliance_rules[rule_name] = rule
    
    def remove_compliance_rule(self, rule_name: str) -> None:
        """Remove a compliance rule."""
        if rule_name in self.compliance_rules:
            del self.compliance_rules[rule_name]
    
    def update_stats(self, stat_name: str, increment: int = 1) -> None:
        """Update compliance statistics."""
        if stat_name in self.validation_stats:
            self.validation_stats[stat_name] += increment
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get compliance validation statistics."""
        return self.validation_stats.copy()


class HIPAAComplianceValidator(ComplianceValidator):
    """HIPAA compliance validator for medical data."""
    
    def __init__(self, level: ComplianceLevel = ComplianceLevel.STANDARD):
        super().__init__(ComplianceStandard.HIPAA, level)
        self.phi_fields = {
            "patient_name", "patient_id", "ssn", "date_of_birth", "address",
            "phone", "email", "medical_record_number", "diagnosis", "treatment"
        }
        self.required_safeguards = {
            "access_control", "audit_logging", "encryption", "authentication"
        }
    
    async def validate_compliance(self, data: Any) -> ComplianceResult:
        """Validate HIPAA compliance of medical data."""
        result = ComplianceResult(
            is_compliant=True,
            standard=self.standard,
            level=self.level
        )
        
        self.update_stats("validations_performed")
        
        # Validate data privacy
        privacy_result = await self.validate_data_privacy(data)
        if not privacy_result.is_compliant:
            result.violations.extend(privacy_result.violations)
            result.is_compliant = False
        
        # Validate security
        security_result = await self.validate_security(data)
        if not security_result.is_compliant:
            result.violations.extend(security_result.violations)
            result.is_compliant = False
        
        # Add warnings and recommendations
        result.warnings.extend(privacy_result.warnings)
        result.warnings.extend(security_result.warnings)
        result.recommendations.extend(privacy_result.recommendations)
        result.recommendations.extend(security_result.recommendations)
        
        # Update stats
        if result.has_violations():
            self.update_stats("compliance_violations")
        if result.has_warnings():
            self.update_stats("compliance_warnings")
        
        return result
    
    async def validate_data_privacy(self, data: Any) -> ComplianceResult:
        """Validate HIPAA data privacy requirements."""
        result = ComplianceResult(
            is_compliant=True,
            standard=self.standard,
            level=self.level
        )
        
        if not isinstance(data, dict):
            result.add_violation("Data must be a dictionary for privacy validation")
            return result
        
        # Check for PHI fields
        phi_found = set()
        for field in self.phi_fields:
            if field in data and data[field] is not None:
                phi_found.add(field)
        
        if phi_found and self.level in [ComplianceLevel.STRICT, ComplianceLevel.AUDIT]:
            # Check if PHI is properly protected
            if not data.get("encryption_level"):
                result.add_violation(f"PHI fields {phi_found} must be encrypted")
            
            if not data.get("access_controls"):
                result.add_violation(f"PHI fields {phi_found} must have access controls")
        
        # Check for minimum necessary principle
        if len(phi_found) > 5 and self.level == ComplianceLevel.AUDIT:
            result.add_warning("Consider if all PHI fields are necessary for this operation")
        
        return result
    
    async def validate_security(self, data: Any) -> ComplianceResult:
        """Validate HIPAA security requirements."""
        result = ComplianceResult(
            is_compliant=True,
            standard=self.standard,
            level=self.level
        )
        
        if not isinstance(data, dict):
            result.add_violation("Data must be a dictionary for security validation")
            return result
        
        # Check for required security safeguards
        safeguards = data.get("security_safeguards", {})
        
        for safeguard in self.required_safeguards:
            if safeguard not in safeguards:
                result.add_violation(f"Missing required security safeguard: {safeguard}")
        
        # Check encryption
        if "encryption" in safeguards:
            encryption = safeguards["encryption"]
            if not encryption.get("algorithm") or encryption.get("algorithm") == "none":
                result.add_violation("Data must be encrypted with a strong algorithm")
            
            if not encryption.get("key_strength") or encryption["key_strength"] < 256:
                result.add_warning("Consider using 256-bit or stronger encryption keys")
        
        # Check authentication
        if "authentication" in safeguards:
            auth = safeguards["authentication"]
            if not auth.get("method") or auth["method"] == "none":
                result.add_violation("Authentication method must be specified")
            
            if auth.get("method") == "password" and not auth.get("mfa_required"):
                result.add_warning("Consider implementing multi-factor authentication")
        
        return result


class GDPRComplianceValidator(ComplianceValidator):
    """GDPR compliance validator for medical data."""
    
    def __init__(self, level: ComplianceLevel = ComplianceLevel.STANDARD):
        super().__init__(ComplianceStandard.GDPR, level)
        self.personal_data_fields = {
            "name", "email", "phone", "address", "date_of_birth", "national_id"
        }
        self.special_categories = {
            "health_data", "genetic_data", "biometric_data"
        }
    
    async def validate_compliance(self, data: Any) -> ComplianceResult:
        """Validate GDPR compliance of medical data."""
        result = ComplianceResult(
            is_compliant=True,
            standard=self.standard,
            level=self.level
        )
        
        self.update_stats("validations_performed")
        
        # Validate data privacy
        privacy_result = await self.validate_data_privacy(data)
        if not privacy_result.is_compliant:
            result.violations.extend(privacy_result.violations)
            result.is_compliant = False
        
        # Validate security
        security_result = await self.validate_security(data)
        if not security_result.is_compliant:
            result.violations.extend(security_result.violations)
            result.is_compliant = False
        
        # Add warnings and recommendations
        result.warnings.extend(privacy_result.warnings)
        result.warnings.extend(security_result.warnings)
        result.recommendations.extend(privacy_result.recommendations)
        result.recommendations.extend(security_result.recommendations)
        
        # Update stats
        if result.has_violations():
            self.update_stats("compliance_violations")
        if result.has_warnings():
            self.update_stats("compliance_warnings")
        
        return result
    
    async def validate_data_privacy(self, data: Any) -> ComplianceResult:
        """Validate GDPR data privacy requirements."""
        result = ComplianceResult(
            is_compliant=True,
            standard=self.standard,
            level=self.level
        )
        
        if not isinstance(data, dict):
            result.add_violation("Data must be a dictionary for privacy validation")
            return result
        
        # Check for legal basis
        if not data.get("legal_basis"):
            result.add_violation("GDPR requires a legal basis for data processing")
        
        # Check for consent if processing special categories
        special_categories_found = set()
        for category in self.special_categories:
            if category in data and data[category] is not None:
                special_categories_found.add(category)
        
        if special_categories_found:
            if not data.get("explicit_consent"):
                result.add_violation("Explicit consent required for special category data")
            
            if not data.get("purpose_limitation"):
                result.add_violation("Purpose limitation must be specified for special category data")
        
        # Check for data minimization
        personal_data_found = set()
        for field in self.personal_data_fields:
            if field in data and data[field] is not None:
                personal_data_found.add(field)
        
        if len(personal_data_found) > 3 and self.level == ComplianceLevel.AUDIT:
            result.add_warning("Consider if all personal data fields are necessary")
        
        return result
    
    async def validate_security(self, data: Any) -> ComplianceResult:
        """Validate GDPR security requirements."""
        result = ComplianceResult(
            is_compliant=True,
            standard=self.standard,
            level=self.level
        )
        
        if not isinstance(data, dict):
            result.add_violation("Data must be a dictionary for security validation")
            return result
        
        # Check for appropriate security measures
        security_measures = data.get("security_measures", {})
        
        if not security_measures.get("encryption"):
            result.add_violation("GDPR requires appropriate encryption measures")
        
        if not security_measures.get("access_controls"):
            result.add_violation("GDPR requires appropriate access controls")
        
        if not security_measures.get("data_retention_policy"):
            result.add_warning("Consider implementing a data retention policy")
        
        return result

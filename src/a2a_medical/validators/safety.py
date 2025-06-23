"""
Medical safety validation for A2A medical systems.

This module provides abstract safety validation frameworks that can be
specialized for different medical domains and safety requirements.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod


class MedicalQueryType(Enum):
    """Types of medical queries."""
    MEDICATION = "medication"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    EMERGENCY = "emergency"
    GENERAL = "general"


class MedicalQuery:
    """Represents a medical query."""
    
    def __init__(self, 
                 query_id: str, 
                 query_text: str, 
                 query_type: MedicalQueryType, 
                 patient_context: Optional[Dict[str, Any]] = None):
        self.query_id = query_id
        self.query_text = query_text
        self.query_type = query_type
        self.patient_context = patient_context or {}


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, 
                 is_safe: bool, 
                 issues: Optional[List[str]] = None, 
                 requires_human_review: bool = False,
                 warnings: Optional[List[str]] = None):
        self.is_safe = is_safe
        self.issues = issues or []
        self.requires_human_review = requires_human_review
        self.warnings = warnings or []


class SafetyIssue:
    """Represents a safety issue found during validation."""
    
    def __init__(self, 
                 severity: str, 
                 category: str, 
                 description: str, 
                 recommendation: str = ""):
        self.severity = severity
        self.category = category
        self.description = description
        self.recommendation = recommendation


class SafetyValidator(ABC):
    """Abstract base class for safety validators.
    
    Provides the foundational structure for implementing medical safety
    validation systems. Concrete implementations should inherit from this class.
    """
    
    def __init__(self, validator_id: str):
        self.validator_id = validator_id
        self.safety_rules: List[Any] = []
        self.risk_thresholds: Dict[str, float] = {}
    
    @abstractmethod
    async def validate_query_safety(self, query: MedicalQuery) -> ValidationResult:
        """Validate the safety of a medical query.
        
        Must be implemented by concrete safety validators.
        """
        pass
    
    @abstractmethod
    async def validate_response_safety(self, response: str, original_query: MedicalQuery) -> ValidationResult:
        """Validate the safety of a medical response.
        
        Must be implemented by concrete safety validators.
        """
        pass
    
    @abstractmethod
    def configure_safety_rules(self, rules: List[Any]) -> None:
        """Configure safety validation rules.
        
        Must be implemented by concrete safety validators.
        """
        pass
    
    @abstractmethod
    def get_safety_guidelines(self) -> Dict[str, List[str]]:
        """Get safety guidelines for this validator.
        
        Must be implemented by concrete safety validators.
        """
        pass
    
    def add_safety_rule(self, rule: Any) -> None:
        """Add a safety rule to this validator."""
        self.safety_rules.append(rule)
    
    def set_risk_threshold(self, category: str, threshold: float) -> None:
        """Set a risk threshold for a specific category."""
        self.risk_thresholds[category] = threshold
    
    def get_validator_info(self) -> Dict[str, Any]:
        """Get information about this safety validator."""
        return {
            "validator_id": self.validator_id,
            "validator_type": self.__class__.__name__,
            "rules_count": len(self.safety_rules),
            "risk_thresholds": self.risk_thresholds.copy()
        }


class MedicalComplianceValidator(ABC):
    """Abstract base class for medical compliance validators.
    
    Provides foundation for implementing compliance validation
    against various medical standards and regulations.
    """
    
    def __init__(self, validator_id: str, compliance_standards: List[str]):
        self.validator_id = validator_id
        self.compliance_standards = compliance_standards
        self.validation_rules: Dict[str, Any] = {}
    
    @abstractmethod
    async def validate_compliance(self, data: Any, standards: List[str]) -> ValidationResult:
        """Validate compliance against specified standards.
        
        Must be implemented by concrete compliance validators.
        """
        pass
    
    @abstractmethod
    def configure_compliance_rules(self, standard: str, rules: Dict[str, Any]) -> None:
        """Configure compliance rules for a specific standard.
        
        Must be implemented by concrete compliance validators.
        """
        pass
    
    @abstractmethod
    def get_compliance_requirements(self, standard: str) -> List[str]:
        """Get compliance requirements for a specific standard.
        
        Must be implemented by concrete compliance validators.
        """
        pass
    
    def add_compliance_standard(self, standard: str) -> None:
        """Add a compliance standard to this validator."""
        if standard not in self.compliance_standards:
            self.compliance_standards.append(standard)
    
    def remove_compliance_standard(self, standard: str) -> None:
        """Remove a compliance standard from this validator."""
        if standard in self.compliance_standards:
            self.compliance_standards.remove(standard)
    
    def get_supported_standards(self) -> List[str]:
        """Get list of supported compliance standards."""
        return self.compliance_standards.copy()


class DrugInteractionValidator(ABC):
    """Abstract base class for drug interaction validators.
    
    Provides foundation for implementing drug interaction checking
    and contraindication validation systems.
    """
    
    def __init__(self, validator_id: str):
        self.validator_id = validator_id
        self.interaction_database: Dict[str, Any] = {}
        self.severity_levels: List[str] = []
    
    @abstractmethod
    async def check_drug_interactions(self, medications: List[str]) -> List[SafetyIssue]:
        """Check for drug interactions among medications.
        
        Must be implemented by concrete drug interaction validators.
        """
        pass
    
    @abstractmethod
    async def validate_contraindications(self, medication: str, conditions: List[str]) -> List[SafetyIssue]:
        """Validate contraindications for a medication given patient conditions.
        
        Must be implemented by concrete drug interaction validators.
        """
        pass
    
    @abstractmethod
    def update_interaction_database(self, data: Dict[str, Any]) -> None:
        """Update the drug interaction database.
        
        Must be implemented by concrete drug interaction validators.
        """
        pass
    
    def get_interaction_info(self, drug1: str, drug2: str) -> Optional[Dict[str, Any]]:
        """Get interaction information between two drugs."""
        key = f"{drug1}_{drug2}" if drug1 < drug2 else f"{drug2}_{drug1}"
        return self.interaction_database.get(key)


class DosageValidator(ABC):
    """Abstract base class for dosage validators.
    
    Provides foundation for implementing medication dosage
    validation and safety checking systems.
    """
    
    def __init__(self, validator_id: str):
        self.validator_id = validator_id
        self.dosage_ranges: Dict[str, Dict[str, float]] = {}
        self.patient_factors: List[str] = []
    
    @abstractmethod
    async def validate_dosage(self, medication: str, dosage: float, patient_info: Dict[str, Any]) -> ValidationResult:
        """Validate medication dosage for a specific patient.
        
        Must be implemented by concrete dosage validators.
        """
        pass
    
    @abstractmethod
    def configure_dosage_ranges(self, medication: str, ranges: Dict[str, float]) -> None:
        """Configure dosage ranges for a medication.
        
        Must be implemented by concrete dosage validators.
        """
        pass
    
    @abstractmethod
    def get_recommended_dosage(self, medication: str, patient_info: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Get recommended dosage for a medication given patient information.
        
        Must be implemented by concrete dosage validators.
        """
        pass
    
    def add_patient_factor(self, factor: str) -> None:
        """Add a patient factor that affects dosage calculations."""
        if factor not in self.patient_factors:
            self.patient_factors.append(factor)


class EmergencyValidator(ABC):
    """Abstract base class for emergency validators.
    
    Provides foundation for implementing emergency condition
    detection and critical situation assessment systems.
    """
    
    def __init__(self, validator_id: str):
        self.validator_id = validator_id
        self.emergency_keywords: List[str] = []
        self.severity_thresholds: Dict[str, float] = {}
    
    @abstractmethod
    async def assess_emergency_level(self, query: MedicalQuery) -> int:
        """Assess the emergency level of a medical query.
        
        Must be implemented by concrete emergency validators.
        """
        pass
    
    @abstractmethod
    async def detect_critical_conditions(self, symptoms: List[str], vitals: Dict[str, float]) -> List[SafetyIssue]:
        """Detect critical medical conditions from symptoms and vitals.
        
        Must be implemented by concrete emergency validators.
        """
        pass
    
    @abstractmethod
    def configure_emergency_criteria(self, criteria: Dict[str, Any]) -> None:
        """Configure criteria for emergency detection.
        
        Must be implemented by concrete emergency validators.
        """
        pass
    
    def add_emergency_keyword(self, keyword: str) -> None:
        """Add an emergency keyword to the detection system."""
        if keyword not in self.emergency_keywords:
            self.emergency_keywords.append(keyword)


# Aliases for backward compatibility
MedicalSafetyValidator = SafetyValidator
ComplianceChecker = MedicalComplianceValidator
ContraindicationChecker = DrugInteractionValidator
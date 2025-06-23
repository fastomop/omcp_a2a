"""
Query validation for medical A2A operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..exceptions import ValidationError


class ValidationLevel(Enum):
    """Validation levels for queries."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPLIANCE = "compliance"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return len(self.warnings) > 0


class QueryValidator(ABC):
    """Base class for query validation in medical A2A systems."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.validation_rules: Dict[str, Any] = {}
        self.validation_stats: Dict[str, int] = {
            "queries_validated": 0,
            "validation_errors": 0,
            "validation_warnings": 0
        }
    
    @abstractmethod
    async def validate_query(self, query: Any) -> ValidationResult:
        """Validate a query and return the result."""
        pass
    
    @abstractmethod
    async def validate_syntax(self, query: Any) -> ValidationResult:
        """Validate query syntax."""
        pass
    
    @abstractmethod
    async def validate_semantics(self, query: Any) -> ValidationResult:
        """Validate query semantics."""
        pass
    
    def add_validation_rule(self, rule_name: str, rule: Any) -> None:
        """Add a validation rule."""
        self.validation_rules[rule_name] = rule
    
    def remove_validation_rule(self, rule_name: str) -> None:
        """Remove a validation rule."""
        if rule_name in self.validation_rules:
            del self.validation_rules[rule_name]
    
    def update_stats(self, stat_name: str, increment: int = 1) -> None:
        """Update validation statistics."""
        if stat_name in self.validation_stats:
            self.validation_stats[stat_name] += increment
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return self.validation_stats.copy()


class MedicalQueryValidator(QueryValidator):
    """Medical-specific query validator."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        self.required_fields = ["query_type", "parameters"]
        self.allowed_query_types = [
            "patient_lookup",
            "medical_record_query",
            "lab_results",
            "medication_history",
            "diagnosis_query"
        ]
    
    async def validate_query(self, query: Any) -> ValidationResult:
        """Validate a medical query comprehensively."""
        result = ValidationResult(is_valid=True, validation_level=self.validation_level)
        
        # Update stats
        self.update_stats("queries_validated")
        
        # Validate syntax
        syntax_result = await self.validate_syntax(query)
        if not syntax_result.is_valid:
            result.errors.extend(syntax_result.errors)
            result.is_valid = False
        
        # Validate semantics
        semantics_result = await self.validate_semantics(query)
        if not semantics_result.is_valid:
            result.errors.extend(semantics_result.errors)
            result.is_valid = False
        
        # Add warnings
        result.warnings.extend(syntax_result.warnings)
        result.warnings.extend(semantics_result.warnings)
        
        # Update stats
        if result.has_errors():
            self.update_stats("validation_errors")
        if result.has_warnings():
            self.update_stats("validation_warnings")
        
        return result
    
    async def validate_syntax(self, query: Any) -> ValidationResult:
        """Validate query syntax."""
        result = ValidationResult(is_valid=True, validation_level=self.validation_level)
        
        if not isinstance(query, dict):
            result.add_error("Query must be a dictionary")
            return result
        
        # Check required fields
        for field in self.required_fields:
            if field not in query:
                result.add_error(f"Missing required field: {field}")
        
        # Check query type
        if "query_type" in query:
            query_type = query["query_type"]
            if not isinstance(query_type, str):
                result.add_error("query_type must be a string")
            elif query_type not in self.allowed_query_types:
                result.add_warning(f"Unknown query type: {query_type}")
        
        # Check parameters
        if "parameters" in query:
            parameters = query["parameters"]
            if not isinstance(parameters, dict):
                result.add_error("parameters must be a dictionary")
        
        return result
    
    async def validate_semantics(self, query: Any) -> ValidationResult:
        """Validate query semantics."""
        result = ValidationResult(is_valid=True, validation_level=self.validation_level)
        
        if not isinstance(query, dict):
            return result
        
        # Validate based on query type
        query_type = query.get("query_type")
        parameters = query.get("parameters", {})
        
        if query_type == "patient_lookup":
            result = await self._validate_patient_lookup(parameters, result)
        elif query_type == "medical_record_query":
            result = await self._validate_medical_record_query(parameters, result)
        elif query_type == "lab_results":
            result = await self._validate_lab_results_query(parameters, result)
        
        return result
    
    async def _validate_patient_lookup(self, parameters: Dict[str, Any], result: ValidationResult) -> ValidationResult:
        """Validate patient lookup query parameters."""
        if not parameters.get("patient_id") and not parameters.get("patient_name"):
            result.add_error("Patient lookup requires either patient_id or patient_name")
        
        return result
    
    async def _validate_medical_record_query(self, parameters: Dict[str, Any], result: ValidationResult) -> ValidationResult:
        """Validate medical record query parameters."""
        if not parameters.get("record_type"):
            result.add_error("Medical record query requires record_type")
        
        return result
    
    async def _validate_lab_results_query(self, parameters: Dict[str, Any], result: ValidationResult) -> ValidationResult:
        """Validate lab results query parameters."""
        if not parameters.get("test_type") and not parameters.get("date_range"):
            result.add_warning("Lab results query should specify test_type or date_range")
        
        return result


class SQLQueryValidator(QueryValidator):
    """SQL query validator for medical databases."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        self.forbidden_keywords = [
            "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE"
        ]
        self.allowed_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY"]
    
    async def validate_query(self, query: Any) -> ValidationResult:
        """Validate an SQL query."""
        result = ValidationResult(is_valid=True, validation_level=self.validation_level)
        
        self.update_stats("queries_validated")
        
        if not isinstance(query, str):
            result.add_error("SQL query must be a string")
            return result
        
        # Basic syntax validation
        syntax_result = await self.validate_syntax(query)
        if not syntax_result.is_valid:
            result.errors.extend(syntax_result.errors)
            result.is_valid = False
        
        # Semantic validation
        semantics_result = await self.validate_semantics(query)
        if not semantics_result.is_valid:
            result.errors.extend(semantics_result.errors)
            result.is_valid = False
        
        result.warnings.extend(syntax_result.warnings)
        result.warnings.extend(semantics_result.warnings)
        
        if result.has_errors():
            self.update_stats("validation_errors")
        if result.has_warnings():
            self.update_stats("validation_warnings")
        
        return result
    
    async def validate_syntax(self, query: str) -> ValidationResult:
        """Validate SQL query syntax."""
        result = ValidationResult(is_valid=True, validation_level=self.validation_level)
        
        if not query.strip():
            result.add_error("Query cannot be empty")
            return result
        
        # Check for forbidden keywords
        query_upper = query.upper()
        for keyword in self.forbidden_keywords:
            if keyword in query_upper:
                result.add_error(f"Forbidden keyword found: {keyword}")
        
        return result
    
    async def validate_semantics(self, query: str) -> ValidationResult:
        """Validate SQL query semantics."""
        result = ValidationResult(is_valid=True, validation_level=self.validation_level)
        
        query_upper = query.upper()
        
        # Check if it's a SELECT query
        if not query_upper.startswith("SELECT"):
            result.add_error("Only SELECT queries are allowed")
        
        # Check for basic structure
        if "FROM" not in query_upper:
            result.add_warning("Query should contain FROM clause")
        
        return result

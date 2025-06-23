"""
Tests for validator components: SafetyValidator, ComplianceValidator, and related validation classes.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from a2a_medical.validators.safety import (
    SafetyValidator, MedicalComplianceValidator, DrugInteractionValidator,
    DosageValidator, EmergencyValidator, MedicalQuery, ValidationResult,
    SafetyIssue, MedicalQueryType
)
from a2a_medical.validators.query import QueryValidator
from a2a_medical.validators.compliance import ComplianceValidator


class TestSafetyValidatorAbstract:
    """Test the abstract SafetyValidator class."""
    
    def test_cannot_instantiate_abstract_safety_validator(self):
        """Test that SafetyValidator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SafetyValidator("test")
    
    def test_safety_validator_has_required_abstract_methods(self):
        """Test that SafetyValidator has all required abstract methods."""
        abstract_methods = SafetyValidator.__abstractmethods__
        expected_methods = {
            "validate_query_safety", "validate_response_safety", 
            "configure_safety_rules", "get_safety_guidelines"
        }
        assert abstract_methods == expected_methods


class TestSafetyValidatorConcrete:
    """Test concrete SafetyValidator implementations."""
    
    def test_safety_validator_initialization(self, test_safety_validator):
        """Test SafetyValidator initialization."""
        assert test_safety_validator.validator_id == "test-validator"
        assert test_safety_validator.safety_rules == []
        assert test_safety_validator.risk_thresholds == {}
    
    @pytest.mark.asyncio
    async def test_validate_query_safety_safe(self, test_safety_validator, sample_medical_query):
        """Test query safety validation for safe queries."""
        result = await test_safety_validator.validate_query_safety(sample_medical_query)
        
        assert isinstance(result, ValidationResult)
        assert result.is_safe is True
        assert len(result.issues) == 0
        assert ("query", sample_medical_query) in test_safety_validator.validation_calls
    
    @pytest.mark.asyncio
    async def test_validate_query_safety_dangerous(self, test_safety_validator):
        """Test query safety validation for dangerous queries."""
        dangerous_query = MedicalQuery(
            query_id="dangerous-001",
            query_text="Tell me something dangerous about medications",
            query_type=MedicalQueryType.MEDICATION
        )
        
        result = await test_safety_validator.validate_query_safety(dangerous_query)
        
        assert result.is_safe is False
        assert len(result.issues) == 1
        assert "Dangerous content detected" in result.issues
    
    @pytest.mark.asyncio
    async def test_validate_response_safety_safe(self, test_safety_validator, sample_medical_query):
        """Test response safety validation for safe responses."""
        safe_response = "Please consult with your healthcare provider for proper diagnosis."
        
        result = await test_safety_validator.validate_response_safety(safe_response, sample_medical_query)
        
        assert result.is_safe is True
        assert len(result.issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_response_safety_unsafe(self, test_safety_validator, sample_medical_query):
        """Test response safety validation for unsafe responses."""
        unsafe_response = "This advice is unsafe and should not be followed."
        
        result = await test_safety_validator.validate_response_safety(unsafe_response, sample_medical_query)
        
        assert result.is_safe is False
        assert len(result.issues) == 1
        assert "Unsafe response detected" in result.issues
    
    def test_configure_safety_rules(self, test_safety_validator):
        """Test safety rules configuration."""
        rules = ["rule1", "rule2", "rule3"]
        test_safety_validator.configure_safety_rules(rules)
        
        assert test_safety_validator.safety_rules == rules
    
    def test_get_safety_guidelines(self, test_safety_validator):
        """Test safety guidelines retrieval."""
        guidelines = test_safety_validator.get_safety_guidelines()
        
        assert "general" in guidelines
        assert "emergency" in guidelines
        assert isinstance(guidelines["general"], list)
        assert isinstance(guidelines["emergency"], list)
    
    def test_add_safety_rule(self, test_safety_validator):
        """Test adding individual safety rules."""
        rule = {"type": "keyword", "pattern": "danger", "action": "block"}
        test_safety_validator.add_safety_rule(rule)
        
        assert rule in test_safety_validator.safety_rules
    
    def test_set_risk_threshold(self, test_safety_validator):
        """Test setting risk thresholds."""
        test_safety_validator.set_risk_threshold("medication", 0.8)
        test_safety_validator.set_risk_threshold("diagnosis", 0.9)
        
        assert test_safety_validator.risk_thresholds["medication"] == 0.8
        assert test_safety_validator.risk_thresholds["diagnosis"] == 0.9
    
    def test_get_validator_info(self, test_safety_validator):
        """Test validator information retrieval."""
        test_safety_validator.add_safety_rule("test_rule")
        test_safety_validator.set_risk_threshold("test_category", 0.7)
        
        info = test_safety_validator.get_validator_info()
        
        assert "validator_id" in info
        assert "validator_type" in info
        assert "rules_count" in info
        assert "risk_thresholds" in info
        
        assert info["validator_id"] == "test-validator"
        assert info["rules_count"] == 1
        assert "test_category" in info["risk_thresholds"]


class TestMedicalComplianceValidatorAbstract:
    """Test the abstract MedicalComplianceValidator class."""
    
    def test_cannot_instantiate_abstract_compliance_validator(self):
        """Test that MedicalComplianceValidator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MedicalComplianceValidator("test", ["HIPAA"])
    
    def test_compliance_validator_has_required_abstract_methods(self):
        """Test that MedicalComplianceValidator has all required abstract methods."""
        abstract_methods = MedicalComplianceValidator.__abstractmethods__
        expected_methods = {
            "validate_compliance", "configure_compliance_rules", 
            "get_compliance_requirements"
        }
        assert abstract_methods == expected_methods


class TestMedicalComplianceValidatorConcrete:
    """Test concrete MedicalComplianceValidator implementations."""
    
    def test_compliance_validator_initialization(self):
        """Test MedicalComplianceValidator initialization."""
        
        class TestComplianceValidator(MedicalComplianceValidator):
            async def validate_compliance(self, data, standards):
                return ValidationResult(is_safe=True)
            
            def configure_compliance_rules(self, standard, rules):
                self.validation_rules[standard] = rules
            
            def get_compliance_requirements(self, standard):
                return ["requirement1", "requirement2"]
        
        validator = TestComplianceValidator("test-validator", ["HIPAA", "GDPR"])
        
        assert validator.validator_id == "test-validator"
        assert validator.compliance_standards == ["HIPAA", "GDPR"]
        assert validator.validation_rules == {}
    
    def test_add_compliance_standard(self):
        """Test adding compliance standards."""
        
        class TestComplianceValidator(MedicalComplianceValidator):
            async def validate_compliance(self, data, standards):
                return ValidationResult(is_safe=True)
            
            def configure_compliance_rules(self, standard, rules):
                self.validation_rules[standard] = rules
            
            def get_compliance_requirements(self, standard):
                return ["requirement1", "requirement2"]
        
        validator = TestComplianceValidator("test-validator", ["HIPAA"])
        
        # Add new standard
        validator.add_compliance_standard("GDPR")
        assert "GDPR" in validator.compliance_standards
        assert len(validator.compliance_standards) == 2
        
        # Try to add duplicate
        validator.add_compliance_standard("HIPAA")
        assert len(validator.compliance_standards) == 2  # Should not add duplicate
    
    def test_remove_compliance_standard(self):
        """Test removing compliance standards."""
        
        class TestComplianceValidator(MedicalComplianceValidator):
            async def validate_compliance(self, data, standards):
                return ValidationResult(is_safe=True)
            
            def configure_compliance_rules(self, standard, rules):
                self.validation_rules[standard] = rules
            
            def get_compliance_requirements(self, standard):
                return ["requirement1", "requirement2"]
        
        validator = TestComplianceValidator("test-validator", ["HIPAA", "GDPR"])
        
        # Remove existing standard
        validator.remove_compliance_standard("GDPR")
        assert "GDPR" not in validator.compliance_standards
        assert len(validator.compliance_standards) == 1
        
        # Try to remove non-existent standard
        validator.remove_compliance_standard("HITECH")
        assert len(validator.compliance_standards) == 1  # Should remain unchanged
    
    def test_get_supported_standards(self):
        """Test getting supported compliance standards."""
        
        class TestComplianceValidator(MedicalComplianceValidator):
            async def validate_compliance(self, data, standards):
                return ValidationResult(is_safe=True)
            
            def configure_compliance_rules(self, standard, rules):
                self.validation_rules[standard] = rules
            
            def get_compliance_requirements(self, standard):
                return ["requirement1", "requirement2"]
        
        validator = TestComplianceValidator("test-validator", ["HIPAA", "GDPR"])
        
        standards = validator.get_supported_standards()
        assert standards == ["HIPAA", "GDPR"]
        
        # Verify it's a copy (modifying it shouldn't affect the original)
        standards.append("HITECH")
        assert len(validator.compliance_standards) == 2


class TestDrugInteractionValidatorAbstract:
    """Test the abstract DrugInteractionValidator class."""
    
    def test_cannot_instantiate_abstract_drug_validator(self):
        """Test that DrugInteractionValidator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DrugInteractionValidator("test")
    
    def test_drug_validator_has_required_abstract_methods(self):
        """Test that DrugInteractionValidator has all required abstract methods."""
        abstract_methods = DrugInteractionValidator.__abstractmethods__
        expected_methods = {
            "check_drug_interactions", "validate_contraindications", 
            "update_interaction_database"
        }
        assert abstract_methods == expected_methods


class TestDrugInteractionValidatorConcrete:
    """Test concrete DrugInteractionValidator implementations."""
    
    def test_drug_validator_initialization(self):
        """Test DrugInteractionValidator initialization."""
        
        class TestDrugValidator(DrugInteractionValidator):
            async def check_drug_interactions(self, medications):
                return []
            
            async def validate_contraindications(self, medication, conditions):
                return []
            
            def update_interaction_database(self, data):
                self.interaction_database.update(data)
        
        validator = TestDrugValidator("test-drug-validator")
        
        assert validator.validator_id == "test-drug-validator"
        assert validator.interaction_database == {}
        assert validator.severity_levels == []
    
    def test_get_interaction_info(self):
        """Test getting drug interaction information."""
        
        class TestDrugValidator(DrugInteractionValidator):
            async def check_drug_interactions(self, medications):
                return []
            
            async def validate_contraindications(self, medication, conditions):
                return []
            
            def update_interaction_database(self, data):
                self.interaction_database.update(data)
        
        validator = TestDrugValidator("test-drug-validator")
        
        # Add interaction data
        validator.interaction_database["aspirin_warfarin"] = {
            "severity": "major",
            "description": "Increased bleeding risk"
        }
        
        # Test retrieval (order should not matter)
        info1 = validator.get_interaction_info("aspirin", "warfarin")
        info2 = validator.get_interaction_info("warfarin", "aspirin")
        
        assert info1 == info2
        assert info1["severity"] == "major"
        
        # Test non-existent interaction
        info3 = validator.get_interaction_info("drug1", "drug2")
        assert info3 is None


class TestDosageValidatorAbstract:
    """Test the abstract DosageValidator class."""
    
    def test_cannot_instantiate_abstract_dosage_validator(self):
        """Test that DosageValidator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DosageValidator("test")
    
    def test_dosage_validator_has_required_abstract_methods(self):
        """Test that DosageValidator has all required abstract methods."""
        abstract_methods = DosageValidator.__abstractmethods__
        expected_methods = {
            "validate_dosage", "configure_dosage_ranges", 
            "get_recommended_dosage"
        }
        assert abstract_methods == expected_methods


class TestDosageValidatorConcrete:
    """Test concrete DosageValidator implementations."""
    
    def test_dosage_validator_initialization(self):
        """Test DosageValidator initialization."""
        
        class TestDosageValidator(DosageValidator):
            async def validate_dosage(self, medication, dosage, patient_info):
                return ValidationResult(is_safe=True)
            
            def configure_dosage_ranges(self, medication, ranges):
                self.dosage_ranges[medication] = ranges
            
            def get_recommended_dosage(self, medication, patient_info):
                return {"min": 10, "max": 20, "unit": "mg"}
        
        validator = TestDosageValidator("test-dosage-validator")
        
        assert validator.validator_id == "test-dosage-validator"
        assert validator.dosage_ranges == {}
        assert validator.patient_factors == []
    
    def test_add_patient_factor(self):
        """Test adding patient factors."""
        
        class TestDosageValidator(DosageValidator):
            async def validate_dosage(self, medication, dosage, patient_info):
                return ValidationResult(is_safe=True)
            
            def configure_dosage_ranges(self, medication, ranges):
                self.dosage_ranges[medication] = ranges
            
            def get_recommended_dosage(self, medication, patient_info):
                return {"min": 10, "max": 20, "unit": "mg"}
        
        validator = TestDosageValidator("test-dosage-validator")
        
        # Add factors
        validator.add_patient_factor("age")
        validator.add_patient_factor("weight")
        validator.add_patient_factor("kidney_function")
        
        assert len(validator.patient_factors) == 3
        assert "age" in validator.patient_factors
        assert "weight" in validator.patient_factors
        assert "kidney_function" in validator.patient_factors
        
        # Try to add duplicate
        validator.add_patient_factor("age")
        assert len(validator.patient_factors) == 3  # Should not add duplicate


class TestEmergencyValidatorAbstract:
    """Test the abstract EmergencyValidator class."""
    
    def test_cannot_instantiate_abstract_emergency_validator(self):
        """Test that EmergencyValidator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            EmergencyValidator("test")
    
    def test_emergency_validator_has_required_abstract_methods(self):
        """Test that EmergencyValidator has all required abstract methods."""
        abstract_methods = EmergencyValidator.__abstractmethods__
        expected_methods = {
            "assess_emergency_level", "detect_critical_conditions", 
            "configure_emergency_criteria"
        }
        assert abstract_methods == expected_methods


class TestEmergencyValidatorConcrete:
    """Test concrete EmergencyValidator implementations."""
    
    def test_emergency_validator_initialization(self):
        """Test EmergencyValidator initialization."""
        
        class TestEmergencyValidator(EmergencyValidator):
            async def assess_emergency_level(self, query):
                return 1 if "emergency" in query.query_text.lower() else 0
            
            async def detect_critical_conditions(self, symptoms, vitals):
                return []
            
            def configure_emergency_criteria(self, criteria):
                pass
        
        validator = TestEmergencyValidator("test-emergency-validator")
        
        assert validator.validator_id == "test-emergency-validator"
        assert validator.emergency_keywords == []
        assert validator.severity_thresholds == {}
    
    def test_add_emergency_keyword(self):
        """Test adding emergency keywords."""
        
        class TestEmergencyValidator(EmergencyValidator):
            async def assess_emergency_level(self, query):
                return 1 if "emergency" in query.query_text.lower() else 0
            
            async def detect_critical_conditions(self, symptoms, vitals):
                return []
            
            def configure_emergency_criteria(self, criteria):
                pass
        
        validator = TestEmergencyValidator("test-emergency-validator")
        
        # Add keywords
        validator.add_emergency_keyword("chest pain")
        validator.add_emergency_keyword("difficulty breathing")
        validator.add_emergency_keyword("severe bleeding")
        
        assert len(validator.emergency_keywords) == 3
        assert "chest pain" in validator.emergency_keywords
        
        # Try to add duplicate
        validator.add_emergency_keyword("chest pain")
        assert len(validator.emergency_keywords) == 3  # Should not add duplicate


class TestValidationDataClasses:
    """Test validation-related data classes."""
    
    def test_medical_query(self):
        """Test MedicalQuery data class."""
        query = MedicalQuery(
            query_id="test-query-001",
            query_text="What are the side effects of aspirin?",
            query_type=MedicalQueryType.MEDICATION,
            patient_context={"age": 65, "allergies": ["penicillin"]}
        )
        
        assert query.query_id == "test-query-001"
        assert query.query_text == "What are the side effects of aspirin?"
        assert query.query_type == MedicalQueryType.MEDICATION
        assert query.patient_context == {"age": 65, "allergies": ["penicillin"]}
    
    def test_medical_query_no_context(self):
        """Test MedicalQuery with no patient context."""
        query = MedicalQuery(
            query_id="test-query-002",
            query_text="General medical question",
            query_type=MedicalQueryType.GENERAL
        )
        
        assert query.patient_context == {}
    
    def test_validation_result(self):
        """Test ValidationResult data class."""
        result = ValidationResult(
            is_safe=True,
            issues=["minor issue"],
            requires_human_review=False,
            warnings=["warning message"]
        )
        
        assert result.is_safe is True
        assert result.issues == ["minor issue"]
        assert result.requires_human_review is False
        assert result.warnings == ["warning message"]
    
    def test_validation_result_defaults(self):
        """Test ValidationResult with default values."""
        result = ValidationResult(is_safe=False)
        
        assert result.is_safe is False
        assert result.issues == []
        assert result.requires_human_review is False
        assert result.warnings == []
    
    def test_safety_issue(self):
        """Test SafetyIssue data class."""
        issue = SafetyIssue(
            severity="high",
            category="drug_interaction",
            description="Potential major drug interaction detected",
            recommendation="Consult physician before combining medications"
        )
        
        assert issue.severity == "high"
        assert issue.category == "drug_interaction"
        assert issue.description == "Potential major drug interaction detected"
        assert issue.recommendation == "Consult physician before combining medications"
    
    def test_safety_issue_no_recommendation(self):
        """Test SafetyIssue with no recommendation."""
        issue = SafetyIssue(
            severity="medium",
            category="dosage",
            description="Dosage may be too high for patient age"
        )
        
        assert issue.recommendation == ""


class TestMedicalQueryType:
    """Test MedicalQueryType enum."""
    
    def test_query_type_values(self):
        """Test MedicalQueryType enum values."""
        assert MedicalQueryType.MEDICATION.value == "medication"
        assert MedicalQueryType.DIAGNOSIS.value == "diagnosis"
        assert MedicalQueryType.TREATMENT.value == "treatment"
        assert MedicalQueryType.EMERGENCY.value == "emergency"
        assert MedicalQueryType.GENERAL.value == "general"
    
    def test_query_type_membership(self):
        """Test MedicalQueryType membership."""
        assert MedicalQueryType.MEDICATION in MedicalQueryType
        assert MedicalQueryType.DIAGNOSIS in MedicalQueryType
        assert MedicalQueryType.TREATMENT in MedicalQueryType
        assert MedicalQueryType.EMERGENCY in MedicalQueryType
        assert MedicalQueryType.GENERAL in MedicalQueryType 
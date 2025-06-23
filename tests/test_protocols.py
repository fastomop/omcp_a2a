"""
Tests for protocol components: MedicalA2AProtocol, AgentDiscovery, MessageRouter, and related protocols.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from a2a_medical.protocols.medical_a2a import (
    MedicalA2AProtocol, ComplianceProtocol, EmergencyProtocol,
    MedicalCapability, EmergencyRequest, EmergencyResponse, ComplianceResult,
    CredentialVerificationResult
)
from a2a_medical.protocols.discovery import (
    AgentDiscovery, MedicalAgentDiscovery, ComplianceChecker, AgentRegistry
)
from a2a_medical.protocols.routing import MessageRouter


class TestMedicalA2AProtocolAbstract:
    """Test the abstract MedicalA2AProtocol class."""
    
    def test_cannot_instantiate_abstract_medical_a2a_protocol(self):
        """Test that MedicalA2AProtocol cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MedicalA2AProtocol("test")
    
    def test_medical_a2a_protocol_has_required_abstract_methods(self):
        """Test that MedicalA2AProtocol has all required abstract methods."""
        abstract_methods = MedicalA2AProtocol.__abstractmethods__
        expected_methods = {
            "validate_medical_request", "handle_medical_emergency", 
            "discover_medical_agents"
        }
        assert abstract_methods == expected_methods


class TestMedicalA2AProtocolConcrete:
    """Test concrete MedicalA2AProtocol implementations."""
    
    def test_medical_a2a_protocol_initialization(self):
        """Test MedicalA2AProtocol initialization."""
        
        class TestMedicalA2AProtocol(MedicalA2AProtocol):
            async def validate_medical_request(self, request):
                return ComplianceResult(is_compliant=True)
            
            async def handle_medical_emergency(self, emergency):
                return EmergencyResponse(status="handled", recommendations=["call_911"])
            
            async def discover_medical_agents(self, capability):
                return []
        
        protocol = TestMedicalA2AProtocol("test-protocol")
        
        assert protocol.protocol_id == "test-protocol"
        assert protocol.medical_capabilities == []
        assert protocol.compliance_validators == []
    
    def test_add_medical_capability(self):
        """Test adding medical capabilities."""
        
        class TestMedicalA2AProtocol(MedicalA2AProtocol):
            async def validate_medical_request(self, request):
                return ComplianceResult(is_compliant=True)
            
            async def handle_medical_emergency(self, emergency):
                return EmergencyResponse(status="handled", recommendations=["call_911"])
            
            async def discover_medical_agents(self, capability):
                return []
        
        protocol = TestMedicalA2AProtocol("test-protocol")
        capability = MedicalCapability("diagnosis", "AI-powered diagnosis", "HIPAA")
        
        protocol.add_medical_capability(capability)
        assert len(protocol.medical_capabilities) == 1
        assert protocol.medical_capabilities[0] == capability
    
    def test_remove_medical_capability(self):
        """Test removing medical capabilities."""
        
        class TestMedicalA2AProtocol(MedicalA2AProtocol):
            async def validate_medical_request(self, request):
                return ComplianceResult(is_compliant=True)
            
            async def handle_medical_emergency(self, emergency):
                return EmergencyResponse(status="handled", recommendations=["call_911"])
            
            async def discover_medical_agents(self, capability):
                return []
        
        protocol = TestMedicalA2AProtocol("test-protocol")
        capability1 = MedicalCapability("diagnosis", "AI diagnosis", "HIPAA")
        capability2 = MedicalCapability("treatment", "Treatment planning", "GDPR")
        
        protocol.add_medical_capability(capability1)
        protocol.add_medical_capability(capability2)
        assert len(protocol.medical_capabilities) == 2
        
        protocol.remove_medical_capability("diagnosis")
        assert len(protocol.medical_capabilities) == 1
        assert protocol.medical_capabilities[0].name == "treatment"
    
    def test_get_medical_capabilities(self):
        """Test getting medical capabilities."""
        
        class TestMedicalA2AProtocol(MedicalA2AProtocol):
            async def validate_medical_request(self, request):
                return ComplianceResult(is_compliant=True)
            
            async def handle_medical_emergency(self, emergency):
                return EmergencyResponse(status="handled", recommendations=["call_911"])
            
            async def discover_medical_agents(self, capability):
                return []
        
        protocol = TestMedicalA2AProtocol("test-protocol")
        capability = MedicalCapability("radiology", "Medical imaging analysis", "HIPAA")
        
        protocol.add_medical_capability(capability)
        capabilities = protocol.get_medical_capabilities()
        
        assert len(capabilities) == 1
        assert capabilities[0].name == "radiology"
        
        # Verify it's a copy
        capabilities.append(MedicalCapability("test", "test", "test"))
        assert len(protocol.medical_capabilities) == 1
    
    def test_add_compliance_validator(self):
        """Test adding compliance validators."""
        
        class TestMedicalA2AProtocol(MedicalA2AProtocol):
            async def validate_medical_request(self, request):
                return ComplianceResult(is_compliant=True)
            
            async def handle_medical_emergency(self, emergency):
                return EmergencyResponse(status="handled", recommendations=["call_911"])
            
            async def discover_medical_agents(self, capability):
                return []
        
        protocol = TestMedicalA2AProtocol("test-protocol")
        validator = Mock()
        
        protocol.add_compliance_validator(validator)
        assert len(protocol.compliance_validators) == 1
        assert protocol.compliance_validators[0] == validator
    
    def test_get_protocol_info(self):
        """Test getting protocol information."""
        
        class TestMedicalA2AProtocol(MedicalA2AProtocol):
            async def validate_medical_request(self, request):
                return ComplianceResult(is_compliant=True)
            
            async def handle_medical_emergency(self, emergency):
                return EmergencyResponse(status="handled", recommendations=["call_911"])
            
            async def discover_medical_agents(self, capability):
                return []
        
        protocol = TestMedicalA2AProtocol("test-protocol")
        protocol.add_medical_capability(MedicalCapability("test", "test", "HIPAA"))
        protocol.add_compliance_validator(Mock())
        
        info = protocol.get_protocol_info()
        
        assert "protocol_id" in info
        assert "protocol_type" in info
        assert "capabilities_count" in info
        assert "validators_count" in info
        
        assert info["protocol_id"] == "test-protocol"
        assert info["capabilities_count"] == 1
        assert info["validators_count"] == 1


class TestComplianceProtocolAbstract:
    """Test the abstract ComplianceProtocol class."""
    
    def test_cannot_instantiate_abstract_compliance_protocol(self):
        """Test that ComplianceProtocol cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ComplianceProtocol("test")
    
    def test_compliance_protocol_has_required_abstract_methods(self):
        """Test that ComplianceProtocol has all required abstract methods."""
        abstract_methods = ComplianceProtocol.__abstractmethods__
        expected_methods = {
            "validate_compliance", "get_compliance_requirements", 
            "configure_validation_rules"
        }
        assert abstract_methods == expected_methods


class TestComplianceProtocolConcrete:
    """Test concrete ComplianceProtocol implementations."""
    
    def test_compliance_protocol_initialization(self):
        """Test ComplianceProtocol initialization."""
        
        class TestComplianceProtocol(ComplianceProtocol):
            async def validate_compliance(self, request):
                return ComplianceResult(is_compliant=True)
            
            def get_compliance_requirements(self):
                return ["requirement1", "requirement2"]
            
            def configure_validation_rules(self, config):
                self.rules = config
        
        protocol = TestComplianceProtocol("HIPAA", "1.0")
        
        assert protocol.standard_name == "HIPAA"
        assert protocol.version == "1.0"
        assert protocol.rules == []
        assert protocol.validators == []
    
    def test_get_standard_info(self):
        """Test getting standard information."""
        
        class TestComplianceProtocol(ComplianceProtocol):
            async def validate_compliance(self, request):
                return ComplianceResult(is_compliant=True)
            
            def get_compliance_requirements(self):
                return ["requirement1", "requirement2"]
            
            def configure_validation_rules(self, config):
                self.rules = config
        
        protocol = TestComplianceProtocol("GDPR", "2.0")
        protocol.rules = ["rule1", "rule2"]
        protocol.validators = [Mock(), Mock()]
        
        info = protocol.get_standard_info()
        
        assert "standard_name" in info
        assert "version" in info
        assert "rules_count" in info
        assert "validators_count" in info
        
        assert info["standard_name"] == "GDPR"
        assert info["version"] == "2.0"
        assert info["rules_count"] == 2
        assert info["validators_count"] == 2


class TestEmergencyProtocolAbstract:
    """Test the abstract EmergencyProtocol class."""
    
    def test_cannot_instantiate_abstract_emergency_protocol(self):
        """Test that EmergencyProtocol cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            EmergencyProtocol("test")
    
    def test_emergency_protocol_has_required_abstract_methods(self):
        """Test that EmergencyProtocol has all required abstract methods."""
        abstract_methods = EmergencyProtocol.__abstractmethods__
        expected_methods = {
            "assess_emergency_level", "handle_emergency", 
            "configure_escalation_rules"
        }
        assert abstract_methods == expected_methods


class TestEmergencyProtocolConcrete:
    """Test concrete EmergencyProtocol implementations."""
    
    def test_emergency_protocol_initialization(self):
        """Test EmergencyProtocol initialization."""
        
        class TestEmergencyProtocol(EmergencyProtocol):
            async def assess_emergency_level(self, request):
                return 1
            
            async def handle_emergency(self, request):
                return EmergencyResponse(status="handled", recommendations=["seek_immediate_care"])
            
            def configure_escalation_rules(self, rules):
                self.escalation_rules = rules
        
        protocol = TestEmergencyProtocol("emergency-protocol")
        
        assert protocol.protocol_name == "emergency-protocol"
        assert protocol.priority_levels == []
        assert protocol.escalation_rules == {}
    
    def test_get_protocol_info(self):
        """Test getting emergency protocol information."""
        
        class TestEmergencyProtocol(EmergencyProtocol):
            async def assess_emergency_level(self, request):
                return 1
            
            async def handle_emergency(self, request):
                return EmergencyResponse(status="handled", recommendations=["seek_immediate_care"])
            
            def configure_escalation_rules(self, rules):
                self.escalation_rules = rules
        
        protocol = TestEmergencyProtocol("test-emergency")
        protocol.priority_levels = [1, 2, 3, 4, 5]
        protocol.escalation_rules = {"level_5": "immediate_response", "level_4": "urgent_response"}
        
        info = protocol.get_protocol_info()
        
        assert "protocol_name" in info
        assert "priority_levels" in info
        assert "escalation_rules_count" in info
        
        assert info["protocol_name"] == "test-emergency"
        assert info["priority_levels"] == [1, 2, 3, 4, 5]
        assert info["escalation_rules_count"] == 2


class TestAgentDiscoveryAbstract:
    """Test the abstract AgentDiscovery class."""
    
    def test_cannot_instantiate_abstract_agent_discovery(self):
        """Test that AgentDiscovery cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AgentDiscovery("test", Mock())
    
    def test_agent_discovery_has_required_abstract_methods(self):
        """Test that AgentDiscovery has all required abstract methods."""
        abstract_methods = AgentDiscovery.__abstractmethods__
        expected_methods = {
            "discover_agent", "discover_multiple_agents", 
            "configure_discovery_strategy"
        }
        assert abstract_methods == expected_methods


class TestAgentRegistryAbstract:
    """Test the abstract AgentRegistry class."""
    
    def test_cannot_instantiate_abstract_agent_registry(self):
        """Test that AgentRegistry cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AgentRegistry("test")
    
    def test_agent_registry_has_required_abstract_methods(self):
        """Test that AgentRegistry has all required abstract methods."""
        abstract_methods = AgentRegistry.__abstractmethods__
        expected_methods = {
            "register_agent", "unregister_agent", "query_agents",
            "get_agent", "list_agents", "get_registry_stats"
        }
        assert abstract_methods == expected_methods


class TestComplianceCheckerAbstract:
    """Test the abstract ComplianceChecker class."""
    
    def test_cannot_instantiate_abstract_compliance_checker(self):
        """Test that ComplianceChecker cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ComplianceChecker("test", ["HIPAA"])
    
    def test_compliance_checker_has_required_abstract_methods(self):
        """Test that ComplianceChecker has all required abstract methods."""
        abstract_methods = ComplianceChecker.__abstractmethods__
        expected_methods = {
            "check_compliance", "configure_compliance_rules", 
            "validate_agent_credentials"
        }
        assert abstract_methods == expected_methods


class TestComplianceCheckerConcrete:
    """Test concrete ComplianceChecker implementations."""
    
    def test_compliance_checker_initialization(self):
        """Test ComplianceChecker initialization."""
        
        class TestComplianceChecker(ComplianceChecker):
            async def check_compliance(self, agent, requirements):
                return True
            
            def configure_compliance_rules(self, standard, rules):
                self.validation_rules[standard] = rules
            
            def validate_agent_credentials(self, agent):
                return True
        
        checker = TestComplianceChecker("test-checker", ["HIPAA", "GDPR"])
        
        assert checker.checker_id == "test-checker"
        assert checker.supported_standards == ["HIPAA", "GDPR"]
        assert checker.validation_rules == {}
    
    def test_add_standard(self):
        """Test adding compliance standards."""
        
        class TestComplianceChecker(ComplianceChecker):
            async def check_compliance(self, agent, requirements):
                return True
            
            def configure_compliance_rules(self, standard, rules):
                self.validation_rules[standard] = rules
            
            def validate_agent_credentials(self, agent):
                return True
        
        checker = TestComplianceChecker("test-checker", ["HIPAA"])
        
        # Add new standard
        checker.add_standard("GDPR")
        assert "GDPR" in checker.supported_standards
        assert len(checker.supported_standards) == 2
        
        # Try to add duplicate
        checker.add_standard("HIPAA")
        assert len(checker.supported_standards) == 2  # Should not add duplicate
    
    def test_get_supported_standards(self):
        """Test getting supported standards."""
        
        class TestComplianceChecker(ComplianceChecker):
            async def check_compliance(self, agent, requirements):
                return True
            
            def configure_compliance_rules(self, standard, rules):
                self.validation_rules[standard] = rules
            
            def validate_agent_credentials(self, agent):
                return True
        
        checker = TestComplianceChecker("test-checker", ["HIPAA", "GDPR"])
        
        standards = checker.get_supported_standards()
        assert standards == ["HIPAA", "GDPR"]
        
        # Verify it's a copy
        standards.append("HITECH")
        assert len(checker.supported_standards) == 2


class TestMedicalAgentDiscoveryAbstract:
    """Test the abstract MedicalAgentDiscovery class."""
    
    def test_cannot_instantiate_abstract_medical_agent_discovery(self):
        """Test that MedicalAgentDiscovery cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MedicalAgentDiscovery("test", Mock())
    
    def test_medical_agent_discovery_has_required_abstract_methods(self):
        """Test that MedicalAgentDiscovery has all required abstract methods."""
        abstract_methods = MedicalAgentDiscovery.__abstractmethods__
        expected_methods = {
            "discover_agent", "discover_multiple_agents", "configure_discovery_strategy",
            "discover_by_medical_specialty", "discover_emergency_agents", 
            "configure_medical_criteria"
        }
        assert abstract_methods == expected_methods


class TestMedicalAgentDiscoveryConcrete:
    """Test concrete MedicalAgentDiscovery implementations."""
    
    def test_medical_agent_discovery_initialization(self):
        """Test MedicalAgentDiscovery initialization."""
        
        class TestMedicalAgentDiscovery(MedicalAgentDiscovery):
            async def discover_agent(self, agent_type, required_capabilities, compliance_requirements=None):
                return None
            
            async def discover_multiple_agents(self, agent_type, required_capabilities, compliance_requirements=None, max_agents=5):
                return []
            
            def configure_discovery_strategy(self, strategy):
                pass
            
            async def discover_by_medical_specialty(self, specialty, compliance_requirements=None):
                return []
            
            async def discover_emergency_agents(self, urgency_level=5):
                return []
            
            def configure_medical_criteria(self, criteria):
                pass
        
        discovery = TestMedicalAgentDiscovery("test-discovery", Mock())
        
        assert discovery.discovery_id == "test-discovery"
        assert discovery.medical_specialties == []
        assert discovery.emergency_capabilities == []
    
    def test_add_medical_specialty(self):
        """Test adding medical specialties."""
        
        class TestMedicalAgentDiscovery(MedicalAgentDiscovery):
            async def discover_agent(self, agent_type, required_capabilities, compliance_requirements=None):
                return None
            
            async def discover_multiple_agents(self, agent_type, required_capabilities, compliance_requirements=None, max_agents=5):
                return []
            
            def configure_discovery_strategy(self, strategy):
                pass
            
            async def discover_by_medical_specialty(self, specialty, compliance_requirements=None):
                return []
            
            async def discover_emergency_agents(self, urgency_level=5):
                return []
            
            def configure_medical_criteria(self, criteria):
                pass
        
        discovery = TestMedicalAgentDiscovery("test-discovery", Mock())
        
        discovery.add_medical_specialty("cardiology")
        discovery.add_medical_specialty("neurology")
        
        assert len(discovery.medical_specialties) == 2
        assert "cardiology" in discovery.medical_specialties
        assert "neurology" in discovery.medical_specialties
        
        # Try to add duplicate
        discovery.add_medical_specialty("cardiology")
        assert len(discovery.medical_specialties) == 2
    
    def test_add_emergency_capability(self):
        """Test adding emergency capabilities."""
        
        class TestMedicalAgentDiscovery(MedicalAgentDiscovery):
            async def discover_agent(self, agent_type, required_capabilities, compliance_requirements=None):
                return None
            
            async def discover_multiple_agents(self, agent_type, required_capabilities, compliance_requirements=None, max_agents=5):
                return []
            
            def configure_discovery_strategy(self, strategy):
                pass
            
            async def discover_by_medical_specialty(self, specialty, compliance_requirements=None):
                return []
            
            async def discover_emergency_agents(self, urgency_level=5):
                return []
            
            def configure_medical_criteria(self, criteria):
                pass
        
        discovery = TestMedicalAgentDiscovery("test-discovery", Mock())
        
        discovery.add_emergency_capability("trauma_response")
        discovery.add_emergency_capability("cardiac_arrest")
        
        assert len(discovery.emergency_capabilities) == 2
        assert "trauma_response" in discovery.emergency_capabilities
        assert "cardiac_arrest" in discovery.emergency_capabilities
        
        # Try to add duplicate
        discovery.add_emergency_capability("trauma_response")
        assert len(discovery.emergency_capabilities) == 2
    
    def test_get_supported_specialties(self):
        """Test getting supported specialties."""
        
        class TestMedicalAgentDiscovery(MedicalAgentDiscovery):
            async def discover_agent(self, agent_type, required_capabilities, compliance_requirements=None):
                return None
            
            async def discover_multiple_agents(self, agent_type, required_capabilities, compliance_requirements=None, max_agents=5):
                return []
            
            def configure_discovery_strategy(self, strategy):
                pass
            
            async def discover_by_medical_specialty(self, specialty, compliance_requirements=None):
                return []
            
            async def discover_emergency_agents(self, urgency_level=5):
                return []
            
            def configure_medical_criteria(self, criteria):
                pass
        
        discovery = TestMedicalAgentDiscovery("test-discovery", Mock())
        discovery.add_medical_specialty("oncology")
        discovery.add_medical_specialty("pediatrics")
        
        specialties = discovery.get_supported_specialties()
        assert specialties == ["oncology", "pediatrics"]
        
        # Verify it's a copy
        specialties.append("dermatology")
        assert len(discovery.medical_specialties) == 2
    
    def test_get_emergency_capabilities(self):
        """Test getting emergency capabilities."""
        
        class TestMedicalAgentDiscovery(MedicalAgentDiscovery):
            async def discover_agent(self, agent_type, required_capabilities, compliance_requirements=None):
                return None
            
            async def discover_multiple_agents(self, agent_type, required_capabilities, compliance_requirements=None, max_agents=5):
                return []
            
            def configure_discovery_strategy(self, strategy):
                pass
            
            async def discover_by_medical_specialty(self, specialty, compliance_requirements=None):
                return []
            
            async def discover_emergency_agents(self, urgency_level=5):
                return []
            
            def configure_medical_criteria(self, criteria):
                pass
        
        discovery = TestMedicalAgentDiscovery("test-discovery", Mock())
        discovery.add_emergency_capability("emergency_surgery")
        discovery.add_emergency_capability("critical_care")
        
        capabilities = discovery.get_emergency_capabilities()
        assert capabilities == ["emergency_surgery", "critical_care"]
        
        # Verify it's a copy
        capabilities.append("trauma_care")
        assert len(discovery.emergency_capabilities) == 2


class TestMessageRouterAbstract:
    """Test the abstract MessageRouter class."""
    
    def test_cannot_instantiate_abstract_message_router(self):
        """Test that MessageRouter cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MessageRouter("test")


class TestProtocolDataClasses:
    """Test protocol-related data classes."""
    
    def test_medical_capability(self):
        """Test MedicalCapability data class."""
        capability = MedicalCapability(
            name="diagnosis",
            description="AI-powered medical diagnosis",
            compliance_level="HIPAA"
        )
        
        assert capability.name == "diagnosis"
        assert capability.description == "AI-powered medical diagnosis"
        assert capability.compliance_level == "HIPAA"
    
    def test_emergency_request(self):
        """Test EmergencyRequest data class."""
        request = EmergencyRequest(
            priority=5,
            description="Chest pain with shortness of breath",
            patient_context={"age": 65, "history": "cardiac"}
        )
        
        assert request.priority == 5
        assert request.description == "Chest pain with shortness of breath"
        assert request.patient_context == {"age": 65, "history": "cardiac"}
    
    def test_emergency_request_no_context(self):
        """Test EmergencyRequest with no patient context."""
        request = EmergencyRequest(
            priority=3,
            description="General emergency"
        )
        
        assert request.priority == 3
        assert request.description == "General emergency"
        assert request.patient_context is None
    
    def test_emergency_response(self):
        """Test EmergencyResponse data class."""
        response = EmergencyResponse(
            status="urgent",
            recommendations=["call_911", "administer_aspirin", "monitor_vitals"],
            escalation_needed=True
        )
        
        assert response.status == "urgent"
        assert response.recommendations == ["call_911", "administer_aspirin", "monitor_vitals"]
        assert response.escalation_needed is True
    
    def test_emergency_response_defaults(self):
        """Test EmergencyResponse with default values."""
        response = EmergencyResponse(
            status="handled",
            recommendations=["rest", "follow_up"]
        )
        
        assert response.status == "handled"
        assert response.recommendations == ["rest", "follow_up"]
        assert response.escalation_needed is False
    
    def test_compliance_result(self):
        """Test ComplianceResult data class."""
        result = ComplianceResult(
            is_compliant=False,
            violations=["missing_encryption", "insufficient_access_control"],
            warnings=["outdated_certificates"]
        )
        
        assert result.is_compliant is False
        assert result.violations == ["missing_encryption", "insufficient_access_control"]
        assert result.warnings == ["outdated_certificates"]
    
    def test_compliance_result_defaults(self):
        """Test ComplianceResult with default values."""
        result = ComplianceResult(is_compliant=True)
        
        assert result.is_compliant is True
        assert result.violations == []
        assert result.warnings == []
    
    def test_credential_verification_result(self):
        """Test CredentialVerificationResult data class."""
        result = CredentialVerificationResult(
            is_verified=True,
            credentials=["medical_license", "board_certification"],
            expires_at=datetime(2024, 12, 31)
        )
        
        assert result.is_verified is True
        assert result.credentials == ["medical_license", "board_certification"]
        assert result.expires_at == datetime(2024, 12, 31)
    
    def test_credential_verification_result_defaults(self):
        """Test CredentialVerificationResult with default values."""
        result = CredentialVerificationResult(is_verified=False)
        
        assert result.is_verified is False
        assert result.credentials == []
        assert result.expires_at is None 
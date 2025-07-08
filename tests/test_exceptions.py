"""
Tests for custom exceptions in the Medical A2A Foundation Framework.
"""

import pytest
from a2a_medical.exceptions import (
    A2AMedicalError, ValidationError, ComplianceError, SecurityError,
    ProtocolError, AgentError, MentalStateError, DiscoveryError, RoutingError
)


class TestA2AMedicalError:
    """Test the base A2AMedicalError exception."""
    
    def test_basic_initialization(self):
        """Test basic A2AMedicalError initialization."""
        error = A2AMedicalError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}
    
    def test_initialization_with_details(self):
        """Test A2AMedicalError initialization with details."""
        details = {"code": "ERR001", "severity": "high"}
        error = A2AMedicalError("Test error with details", details)
        
        assert str(error) == "Test error with details"
        assert error.message == "Test error with details"
        assert error.details == details
        assert error.details["code"] == "ERR001"
        assert error.details["severity"] == "high"
    
    def test_inheritance(self):
        """Test that A2AMedicalError inherits from Exception."""
        error = A2AMedicalError("Test error")
        assert isinstance(error, Exception)
        assert isinstance(error, A2AMedicalError)
    
    def test_empty_details_default(self):
        """Test that details defaults to empty dict when None is passed."""
        error = A2AMedicalError("Test error", None)
        assert error.details == {}


class TestValidationError:
    """Test the ValidationError exception."""
    
    def test_basic_initialization(self):
        """Test basic ValidationError initialization."""
        error = ValidationError("Invalid input")
        
        assert str(error) == "Invalid input"
        assert error.message == "Invalid input"
        assert error.field is None
        assert error.value is None
        assert error.details == {"field": None, "value": None}
    
    def test_initialization_with_field_and_value(self):
        """Test ValidationError initialization with field and value."""
        error = ValidationError("Age must be positive", "age", -5)
        
        assert str(error) == "Age must be positive"
        assert error.message == "Age must be positive"
        assert error.field == "age"
        assert error.value == -5
        assert error.details == {"field": "age", "value": -5}
    
    def test_inheritance(self):
        """Test that ValidationError inherits from A2AMedicalError."""
        error = ValidationError("Validation failed")
        assert isinstance(error, A2AMedicalError)
        assert isinstance(error, ValidationError)
        assert isinstance(error, Exception)
    
    def test_with_complex_value(self):
        """Test ValidationError with complex value types."""
        complex_value = {"nested": {"data": [1, 2, 3]}}
        error = ValidationError("Complex validation failed", "nested_field", complex_value)
        
        assert error.field == "nested_field"
        assert error.value == complex_value
        assert error.details["value"]["nested"]["data"] == [1, 2, 3]


class TestComplianceError:
    """Test the ComplianceError exception."""
    
    def test_basic_initialization(self):
        """Test basic ComplianceError initialization."""
        error = ComplianceError("Compliance violation")
        
        assert str(error) == "Compliance violation"
        assert error.message == "Compliance violation"
        assert error.regulation is None
        assert error.requirement is None
        assert error.details == {"regulation": None, "requirement": None}
    
    def test_initialization_with_regulation_and_requirement(self):
        """Test ComplianceError initialization with regulation and requirement."""
        error = ComplianceError("HIPAA violation", "HIPAA", "data_encryption")
        
        assert str(error) == "HIPAA violation"
        assert error.regulation == "HIPAA"
        assert error.requirement == "data_encryption"
        assert error.details == {"regulation": "HIPAA", "requirement": "data_encryption"}
    
    def test_inheritance(self):
        """Test that ComplianceError inherits from A2AMedicalError."""
        error = ComplianceError("Compliance failed")
        assert isinstance(error, A2AMedicalError)
        assert isinstance(error, ComplianceError)
    
    def test_gdpr_compliance_example(self):
        """Test ComplianceError with GDPR example."""
        error = ComplianceError("GDPR consent not obtained", "GDPR", "explicit_consent")
        
        assert error.regulation == "GDPR"
        assert error.requirement == "explicit_consent"


class TestSecurityError:
    """Test the SecurityError exception."""
    
    def test_basic_initialization(self):
        """Test basic SecurityError initialization."""
        error = SecurityError("Security breach detected")
        
        assert str(error) == "Security breach detected"
        assert error.security_level is None
        assert error.violation_type is None
        assert error.details == {"security_level": None, "violation_type": None}
    
    def test_initialization_with_security_details(self):
        """Test SecurityError initialization with security details."""
        error = SecurityError("Unauthorized access", "high", "authentication_failure")
        
        assert str(error) == "Unauthorized access"
        assert error.security_level == "high"
        assert error.violation_type == "authentication_failure"
        assert error.details == {"security_level": "high", "violation_type": "authentication_failure"}
    
    def test_inheritance(self):
        """Test that SecurityError inherits from A2AMedicalError."""
        error = SecurityError("Security failed")
        assert isinstance(error, A2AMedicalError)
        assert isinstance(error, SecurityError)
    
    def test_encryption_security_example(self):
        """Test SecurityError with encryption example."""
        error = SecurityError("Encryption key compromised", "critical", "key_exposure")
        
        assert error.security_level == "critical"
        assert error.violation_type == "key_exposure"


class TestProtocolError:
    """Test the ProtocolError exception."""
    
    def test_basic_initialization(self):
        """Test basic ProtocolError initialization."""
        error = ProtocolError("Protocol communication failed")
        
        assert str(error) == "Protocol communication failed"
        assert error.protocol is None
        assert error.endpoint is None
        assert error.details == {"protocol": None, "endpoint": None}
    
    def test_initialization_with_protocol_details(self):
        """Test ProtocolError initialization with protocol details."""
        error = ProtocolError("A2A handshake failed", "A2A-v1.0", "/api/handshake")
        
        assert str(error) == "A2A handshake failed"
        assert error.protocol == "A2A-v1.0"
        assert error.endpoint == "/api/handshake"
        assert error.details == {"protocol": "A2A-v1.0", "endpoint": "/api/handshake"}
    
    def test_inheritance(self):
        """Test that ProtocolError inherits from A2AMedicalError."""
        error = ProtocolError("Protocol failed")
        assert isinstance(error, A2AMedicalError)
        assert isinstance(error, ProtocolError)
    
    def test_http_protocol_example(self):
        """Test ProtocolError with HTTP example."""
        error = ProtocolError("HTTP timeout", "HTTP/1.1", "https://api.medical.com/v1/agents")
        
        assert error.protocol == "HTTP/1.1"
        assert error.endpoint == "https://api.medical.com/v1/agents"


class TestAgentError:
    """Test the AgentError exception."""
    
    def test_basic_initialization(self):
        """Test basic AgentError initialization."""
        error = AgentError("Agent operation failed")
        
        assert str(error) == "Agent operation failed"
        assert error.agent_id is None
        assert error.operation is None
        assert error.details == {"agent_id": None, "operation": None}
    
    def test_initialization_with_agent_details(self):
        """Test AgentError initialization with agent details."""
        error = AgentError("Agent startup failed", "medical-agent-001", "initialize")
        
        assert str(error) == "Agent startup failed"
        assert error.agent_id == "medical-agent-001"
        assert error.operation == "initialize"
        assert error.details == {"agent_id": "medical-agent-001", "operation": "initialize"}
    
    def test_inheritance(self):
        """Test that AgentError inherits from A2AMedicalError."""
        error = AgentError("Agent failed")
        assert isinstance(error, A2AMedicalError)
        assert isinstance(error, AgentError)
    
    def test_agent_communication_example(self):
        """Test AgentError with communication example."""
        error = AgentError("Communication timeout", "dr-watson-ai", "send_message")
        
        assert error.agent_id == "dr-watson-ai"
        assert error.operation == "send_message"


class TestMentalStateError:
    """Test the MentalStateError exception."""
    
    def test_basic_initialization(self):
        """Test basic MentalStateError initialization."""
        error = MentalStateError("Mental state update failed")
        
        assert str(error) == "Mental state update failed"
        assert error.state_type is None
        assert error.transition is None
        assert error.details == {"state_type": None, "transition": None}
    
    def test_initialization_with_state_details(self):
        """Test MentalStateError initialization with state details."""
        error = MentalStateError("Invalid state transition", "emotion", "calm_to_panic")
        
        assert str(error) == "Invalid state transition"
        assert error.state_type == "emotion"
        assert error.transition == "calm_to_panic"
        assert error.details == {"state_type": "emotion", "transition": "calm_to_panic"}
    
    def test_inheritance(self):
        """Test that MentalStateError inherits from A2AMedicalError."""
        error = MentalStateError("Mental state failed")
        assert isinstance(error, A2AMedicalError)
        assert isinstance(error, MentalStateError)
    
    def test_belief_state_example(self):
        """Test MentalStateError with belief state example."""
        error = MentalStateError("Belief update failed", "medical_belief", "uncertain_to_confident")
        
        assert error.state_type == "medical_belief"
        assert error.transition == "uncertain_to_confident"


class TestDiscoveryError:
    """Test the DiscoveryError exception."""
    
    def test_basic_initialization(self):
        """Test basic DiscoveryError initialization."""
        error = DiscoveryError("Agent discovery failed")
        
        assert str(error) == "Agent discovery failed"
        assert error.discovery_method is None
        assert error.target is None
        assert error.details == {"discovery_method": None, "target": None}
    
    def test_initialization_with_discovery_details(self):
        """Test DiscoveryError initialization with discovery details."""
        error = DiscoveryError("DNS discovery timeout", "dns_sd", "medical-agents.local")
        
        assert str(error) == "DNS discovery timeout"
        assert error.discovery_method == "dns_sd"
        assert error.target == "medical-agents.local"
        assert error.details == {"discovery_method": "dns_sd", "target": "medical-agents.local"}
    
    def test_inheritance(self):
        """Test that DiscoveryError inherits from A2AMedicalError."""
        error = DiscoveryError("Discovery failed")
        assert isinstance(error, A2AMedicalError)
        assert isinstance(error, DiscoveryError)
    
    def test_registry_discovery_example(self):
        """Test DiscoveryError with registry example."""
        error = DiscoveryError("Registry unreachable", "registry_lookup", "https://registry.medical.ai")
        
        assert error.discovery_method == "registry_lookup"
        assert error.target == "https://registry.medical.ai"


class TestRoutingError:
    """Test the RoutingError exception."""
    
    def test_basic_initialization(self):
        """Test basic RoutingError initialization."""
        error = RoutingError("Message routing failed")
        
        assert str(error) == "Message routing failed"
        assert error.route is None
        assert error.destination is None
        assert error.details == {"route": None, "destination": None}
    
    def test_initialization_with_routing_details(self):
        """Test RoutingError initialization with routing details."""
        error = RoutingError("Route not found", "emergency_protocol", "trauma-center-001")
        
        assert str(error) == "Route not found"
        assert error.route == "emergency_protocol"
        assert error.destination == "trauma-center-001"
        assert error.details == {"route": "emergency_protocol", "destination": "trauma-center-001"}
    
    def test_inheritance(self):
        """Test that RoutingError inherits from A2AMedicalError."""
        error = RoutingError("Routing failed")
        assert isinstance(error, A2AMedicalError)
        assert isinstance(error, RoutingError)
    
    def test_load_balancer_example(self):
        """Test RoutingError with load balancer example."""
        error = RoutingError("Load balancer unreachable", "round_robin", "medical-cluster-west")
        
        assert error.route == "round_robin"
        assert error.destination == "medical-cluster-west"


class TestExceptionInteroperability:
    """Test exception interoperability and chaining."""
    
    def test_exception_chaining(self):
        """Test exception chaining with cause."""
        original_error = ValueError("Original validation error")
        
        try:
            raise ValidationError("Field validation failed", "email", "invalid@") from original_error
        except ValidationError as e:
            assert e.field == "email"
            assert e.value == "invalid@"
            assert e.__cause__ == original_error
            assert isinstance(e.__cause__, ValueError)
    
    def test_multiple_exception_types(self):
        """Test that different exception types can be distinguished."""
        validation_error = ValidationError("Invalid data")
        security_error = SecurityError("Security breach")
        protocol_error = ProtocolError("Protocol failure")
        
        assert type(validation_error) != type(security_error)
        assert type(security_error) != type(protocol_error)
        
        # All should be A2AMedicalError instances
        assert isinstance(validation_error, A2AMedicalError)
        assert isinstance(security_error, A2AMedicalError)
        assert isinstance(protocol_error, A2AMedicalError)
    
    def test_exception_details_accessibility(self):
        """Test that exception details are accessible from base class."""
        error = ComplianceError("HIPAA violation", "HIPAA", "encryption_required")
        
        # Should be accessible both ways
        assert error.regulation == "HIPAA"
        assert error.details["regulation"] == "HIPAA"
        assert error.requirement == "encryption_required"
        assert error.details["requirement"] == "encryption_required"
    
    def test_exception_serialization_compatibility(self):
        """Test that exceptions can be converted to strings and contain useful info."""
        error = AgentError("Agent communication timeout", "medical-ai-001", "heartbeat")
        
        error_str = str(error)
        assert "Agent communication timeout" in error_str
        
        # Details should be accessible for logging/debugging
        assert error.details["agent_id"] == "medical-ai-001"
        assert error.details["operation"] == "heartbeat"


class TestExceptionRaising:
    """Test that exceptions can be properly raised and caught."""
    
    def test_raise_and_catch_validation_error(self):
        """Test raising and catching ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test validation error", "test_field", "test_value")
        
        error = exc_info.value
        assert error.field == "test_field"
        assert error.value == "test_value"
    
    def test_raise_and_catch_compliance_error(self):
        """Test raising and catching ComplianceError."""
        with pytest.raises(ComplianceError) as exc_info:
            raise ComplianceError("Test compliance error", "TEST_REG", "test_requirement")
        
        error = exc_info.value
        assert error.regulation == "TEST_REG"
        assert error.requirement == "test_requirement"
    
    def test_catch_base_exception(self):
        """Test that specific exceptions can be caught as base A2AMedicalError."""
        with pytest.raises(A2AMedicalError) as exc_info:
            raise SecurityError("Security test error", "high", "test_violation")
        
        # Should catch as base class but maintain specific type
        error = exc_info.value
        assert isinstance(error, SecurityError)
        assert isinstance(error, A2AMedicalError)
        assert error.security_level == "high" 
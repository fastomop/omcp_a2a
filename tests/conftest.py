"""
Pytest configuration and shared fixtures for A2A Medical Framework tests.
"""

import pytest
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
import uuid

# Import the framework components for testing
from a2a_medical.base.agent import (
    WorldModel, MedicalAgent, MentalState, CognitiveModule,
    ProcessedObservation, Action, ActionResult
)
from a2a_medical.protocols.medical_a2a import (
    MedicalA2AProtocol, ComplianceProtocol, EmergencyProtocol,
    MedicalCapability, EmergencyRequest, EmergencyResponse, ComplianceResult
)
from a2a_medical.validators.safety import (
    SafetyValidator, MedicalComplianceValidator, DrugInteractionValidator,
    DosageValidator, EmergencyValidator, MedicalQuery, ValidationResult,
    SafetyIssue, MedicalQueryType
)
from a2a_medical.utils.logging import (
    MedicalLogger, ComplianceLogger, PHIDetector, LogSanitizer,
    LogEntry, AuditEvent
)
from a2a_medical.utils.metrics import (
    MetricsCollector, MedicalMetricsCollector, PerformanceMonitor,
    ComplianceMonitor, MetricType, MetricValue, ComplianceMetric
)


# ================== Concrete Test Implementations ==================

class TestWorldModel(WorldModel):
    """Concrete implementation of WorldModel for testing."""
    
    def __init__(self):
        super().__init__()
        self.knowledge_base = {}
        self.predictions = {}
    
    def update(self, observation: ProcessedObservation) -> None:
        self.knowledge_base[observation.source] = observation.data
        self.last_updated = observation.timestamp
    
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        return self.knowledge_base.get(query, "No information available")
    
    def predict(self, scenario: Dict[str, Any]) -> Any:
        scenario_key = str(scenario)
        return self.predictions.get(scenario_key, {"prediction": "unknown", "confidence": 0.5})
    
    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "knowledge_items": len(self.knowledge_base),
            "predictions_cached": len(self.predictions),
            "last_updated": self.last_updated
        }
    
    def reset(self) -> None:
        self.knowledge_base.clear()
        self.predictions.clear()
        self.last_updated = None


class TestMedicalAgent(MedicalAgent):
    """Concrete implementation of MedicalAgent for testing."""
    
    def __init__(self, agent_id: str = "test-agent", world_model: Optional[WorldModel] = None):
        if world_model is None:
            world_model = TestWorldModel()
        super().__init__(
            agent_id=agent_id,
            agent_type="test",
            capabilities=["testing", "validation"],
            world_model=world_model
        )
        self.perception_calls = []
        self.learning_calls = []
        self.reasoning_calls = []
        self.execution_calls = []
    
    async def perceive(self, observation: Any) -> ProcessedObservation:
        self.perception_calls.append(observation)
        return ProcessedObservation(
            data=observation,
            timestamp=datetime.now().timestamp(),
            source="test_input",
            confidence=0.9
        )
    
    async def learn(self, state: MentalState, observation: ProcessedObservation) -> MentalState:
        self.learning_calls.append((state, observation))
        state.update_world_model(observation)
        return state
    
    async def reason(self, state: MentalState) -> Action:
        self.reasoning_calls.append(state)
        return Action(
            action_type="test_action",
            parameters={"test": "data"},
            priority=1
        )
    
    async def execute(self, action: Action) -> ActionResult:
        self.execution_calls.append(action)
        return ActionResult(
            success=True,
            data={"executed": action.action_type},
            metadata=action.parameters
        )
    
    def build_agent_card(self):
        # Mock implementation since we can't import A2A types in tests
        return Mock()


class TestSafetyValidator(SafetyValidator):
    """Concrete implementation of SafetyValidator for testing."""
    
    def __init__(self, validator_id: str = "test-validator"):
        super().__init__(validator_id)
        self.validation_calls = []
    
    async def validate_query_safety(self, query: MedicalQuery) -> ValidationResult:
        self.validation_calls.append(("query", query))
        # Simple test logic
        is_safe = "dangerous" not in query.query_text.lower()
        issues = [] if is_safe else ["Dangerous content detected"]
        return ValidationResult(is_safe=is_safe, issues=issues)
    
    async def validate_response_safety(self, response: str, original_query: MedicalQuery) -> ValidationResult:
        self.validation_calls.append(("response", response, original_query))
        is_safe = "unsafe" not in response.lower()
        issues = [] if is_safe else ["Unsafe response detected"]
        return ValidationResult(is_safe=is_safe, issues=issues)
    
    def configure_safety_rules(self, rules: List[Any]) -> None:
        self.safety_rules = rules
    
    def get_safety_guidelines(self) -> Dict[str, List[str]]:
        return {
            "general": ["Always prioritize patient safety", "Validate all medical advice"],
            "emergency": ["Direct to emergency services", "Do not delay care"]
        }


class TestMedicalLogger(MedicalLogger):
    """Concrete implementation of MedicalLogger for testing."""
    
    def __init__(self, logger_id: str = "test-logger"):
        super().__init__(logger_id)
        self.logged_events = []
        self.audit_events = []
    
    def log_medical_event(self, level: str, message: str, agent_id: Optional[str] = None,
                         patient_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.logged_events.append({
            "level": level,
            "message": message,
            "agent_id": agent_id,
            "patient_id": patient_id,
            "metadata": metadata,
            "timestamp": datetime.now()
        })
    
    def log_audit_event(self, audit_event: AuditEvent) -> None:
        self.audit_events.append(audit_event)
    
    def configure_phi_protection(self, config: Dict[str, Any]) -> None:
        self.phi_config = config
    
    def search_logs(self, criteria: Dict[str, Any], start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[LogEntry]:
        # Simple mock search
        return [LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            message="Test log entry",
            metadata=criteria
        )]
    
    def export_audit_trail(self, start_time: datetime, end_time: datetime, format: str = "json") -> str:
        return f"Audit trail from {start_time} to {end_time} in {format} format"


class TestMetricsCollector(MetricsCollector):
    """Concrete implementation of MetricsCollector for testing."""
    
    def __init__(self, collector_id: str = "test-collector"):
        super().__init__(collector_id)
        self.collected_metrics = []
    
    def collect_metric(self, metric_name: str, value: float, metric_type: MetricType,
                      labels: Optional[Dict[str, str]] = None) -> None:
        self.collected_metrics.append({
            "name": metric_name,
            "value": value,
            "type": metric_type,
            "labels": labels or {},
            "timestamp": datetime.now()
        })
    
    def get_metric_values(self, metric_name: str, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[MetricValue]:
        matching_metrics = [m for m in self.collected_metrics if m["name"] == metric_name]
        return [
            MetricValue(
                value=m["value"],
                timestamp=m["timestamp"],
                labels=m["labels"],
                metric_type=m["type"]
            ) for m in matching_metrics
        ]
    
    def aggregate_metrics(self, metric_name: str, aggregation_type: str, time_window: timedelta) -> Optional[float]:
        values = [m["value"] for m in self.collected_metrics if m["name"] == metric_name]
        if not values:
            return None
        if aggregation_type == "sum":
            return sum(values)
        elif aggregation_type == "avg":
            return sum(values) / len(values)
        elif aggregation_type == "max":
            return max(values)
        elif aggregation_type == "min":
            return min(values)
        return None
    
    def export_metrics(self, format: str = "prometheus") -> str:
        return f"Exported {len(self.collected_metrics)} metrics in {format} format"
    
    def configure_retention(self, metric_name: str, retention_period: timedelta) -> None:
        pass


# ================== Pytest Fixtures ==================

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_world_model():
    """Provide a test world model instance."""
    return TestWorldModel()


@pytest.fixture
def test_medical_agent(test_world_model):
    """Provide a test medical agent instance."""
    return TestMedicalAgent(world_model=test_world_model)


@pytest.fixture
def test_safety_validator():
    """Provide a test safety validator instance."""
    return TestSafetyValidator()


@pytest.fixture
def test_medical_logger():
    """Provide a test medical logger instance."""
    return TestMedicalLogger()


@pytest.fixture
def test_metrics_collector():
    """Provide a test metrics collector instance."""
    return TestMetricsCollector()


@pytest.fixture
def sample_medical_query():
    """Provide a sample medical query for testing."""
    return MedicalQuery(
        query_id="test-query-001",
        query_text="What are the symptoms of diabetes?",
        query_type=MedicalQueryType.GENERAL,
        patient_context={"age": 45, "gender": "female"}
    )


@pytest.fixture
def sample_processed_observation():
    """Provide a sample processed observation for testing."""
    return ProcessedObservation(
        data={"symptom": "fever", "severity": "mild"},
        timestamp=datetime.now().timestamp(),
        source="patient_input",
        confidence=0.8
    )


@pytest.fixture
def sample_action():
    """Provide a sample action for testing."""
    return Action(
        action_type="provide_recommendation",
        parameters={"recommendation": "consult physician"},
        priority=2,
        metadata={"urgency": "low"}
    )


@pytest.fixture
def sample_audit_event():
    """Provide a sample audit event for testing."""
    return AuditEvent(
        event_id=str(uuid.uuid4()),
        event_type="data_access",
        timestamp=datetime.now(),
        actor="test_user",
        resource="patient_record",
        action="read",
        outcome="success"
    )


@pytest.fixture
def sample_compliance_metric():
    """Provide a sample compliance metric for testing."""
    return ComplianceMetric(
        standard="HIPAA",
        compliance_level=0.95,
        violations_count=2,
        timestamp=datetime.now(),
        details={"violations": ["minor_logging_issue", "access_delay"]}
    )


# ================== Mock Fixtures ==================

@pytest.fixture
def mock_a2a_client():
    """Provide a mock A2A client for testing."""
    client = Mock()
    client.send_message = AsyncMock()
    client.receive_message = AsyncMock()
    return client


@pytest.fixture
def mock_agent_card():
    """Provide a mock agent card for testing."""
    card = Mock()
    card.name = "test-agent-001"
    card.description = "Test medical agent"
    card.version = "1.0.0"
    card.capabilities = Mock()
    card.skills = []
    return card


# ================== Utility Functions ==================

def assert_abstract_method_raises(abstract_class, method_name, *args, **kwargs):
    """Helper function to test that abstract methods raise TypeError when called."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        instance = abstract_class()


def create_test_instance(abstract_class, implementations: Dict[str, Any]):
    """Helper to create test instances of abstract classes with provided method implementations."""
    class TestImplementation(abstract_class):
        pass
    
    for method_name, implementation in implementations.items():
        setattr(TestImplementation, method_name, implementation)
    
    return TestImplementation


# ================== Test Markers ==================

# Custom pytest markers for organizing tests
pytest_markers = [
    "unit: Unit tests for individual components",
    "integration: Integration tests for component interactions", 
    "compliance: Compliance and regulatory tests",
    "security: Security-focused tests",
    "performance: Performance and load tests",
    "async: Tests for asynchronous functionality",
    "mock: Tests using mocked dependencies"
] 
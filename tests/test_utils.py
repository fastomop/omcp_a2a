"""
Tests for utility components: MedicalLogger, MetricsCollector, CryptoManager, and related utilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from a2a_medical.utils.logging import (
    MedicalLogger, ComplianceLogger, PHIDetector, LogSanitizer,
    SecureLogger, LogEntry, AuditEvent
)
from a2a_medical.utils.metrics import (
    MetricsCollector, MedicalMetricsCollector, PerformanceMonitor,
    ComplianceMonitor, SecurityMetricsCollector, MetricType, MetricValue,
    PerformanceMetric, ComplianceMetric
)
from a2a_medical.utils.crypto import CryptoManager


class TestMedicalLoggerAbstract:
    """Test the abstract MedicalLogger class."""
    
    def test_cannot_instantiate_abstract_medical_logger(self):
        """Test that MedicalLogger cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MedicalLogger("test")
    
    def test_medical_logger_has_required_abstract_methods(self):
        """Test that MedicalLogger has all required abstract methods."""
        abstract_methods = MedicalLogger.__abstractmethods__
        expected_methods = {
            "log_medical_event", "log_audit_event", "configure_phi_protection",
            "search_logs", "export_audit_trail"
        }
        assert abstract_methods == expected_methods


class TestMedicalLoggerConcrete:
    """Test concrete MedicalLogger implementations."""
    
    def test_medical_logger_initialization(self, test_medical_logger):
        """Test MedicalLogger initialization."""
        assert test_medical_logger.logger_id == "test-logger"
        assert test_medical_logger.phi_protection is True
        assert test_medical_logger.audit_enabled is True
        assert test_medical_logger.phi_detector is None
        assert test_medical_logger.sanitizer is None
    
    def test_log_medical_event(self, test_medical_logger):
        """Test medical event logging."""
        test_medical_logger.log_medical_event(
            level="INFO",
            message="Patient record accessed",
            agent_id="agent-001",
            patient_id="patient-123",
            metadata={"action": "read", "timestamp": datetime.now()}
        )
        
        assert len(test_medical_logger.logged_events) == 1
        event = test_medical_logger.logged_events[0]
        assert event["level"] == "INFO"
        assert event["message"] == "Patient record accessed"
        assert event["agent_id"] == "agent-001"
        assert event["patient_id"] == "patient-123"
        assert "timestamp" in event
    
    def test_log_audit_event(self, test_medical_logger, sample_audit_event):
        """Test audit event logging."""
        test_medical_logger.log_audit_event(sample_audit_event)
        
        assert len(test_medical_logger.audit_events) == 1
        assert test_medical_logger.audit_events[0] == sample_audit_event
    
    def test_configure_phi_protection(self, test_medical_logger):
        """Test PHI protection configuration."""
        config = {
            "detection_level": "strict",
            "sanitization_mode": "mask",
            "encryption": True
        }
        
        test_medical_logger.configure_phi_protection(config)
        assert hasattr(test_medical_logger, 'phi_config')
        assert test_medical_logger.phi_config == config
    
    def test_search_logs(self, test_medical_logger):
        """Test log searching functionality."""
        criteria = {"level": "ERROR", "agent_id": "agent-001"}
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        results = test_medical_logger.search_logs(criteria, start_time, end_time)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], LogEntry)
    
    def test_export_audit_trail(self, test_medical_logger):
        """Test audit trail export."""
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        
        audit_trail = test_medical_logger.export_audit_trail(start_time, end_time, "json")
        
        assert isinstance(audit_trail, str)
        assert "json" in audit_trail.lower()
    
    def test_phi_protection_controls(self, test_medical_logger):
        """Test PHI protection enable/disable controls."""
        # Test enable
        test_medical_logger.enable_phi_protection()
        assert test_medical_logger.phi_protection is True
        
        # Test disable
        test_medical_logger.disable_phi_protection()
        assert test_medical_logger.phi_protection is False
    
    def test_audit_controls(self, test_medical_logger):
        """Test audit enable/disable controls."""
        # Test enable
        test_medical_logger.enable_audit()
        assert test_medical_logger.audit_enabled is True
        
        # Test disable
        test_medical_logger.disable_audit()
        assert test_medical_logger.audit_enabled is False
    
    def test_get_logger_info(self, test_medical_logger):
        """Test logger information retrieval."""
        test_medical_logger.enable_phi_protection()
        test_medical_logger.enable_audit()
        
        info = test_medical_logger.get_logger_info()
        
        assert "logger_id" in info
        assert "logger_type" in info
        assert "phi_protection" in info
        assert "audit_enabled" in info
        
        assert info["logger_id"] == "test-logger"
        assert info["phi_protection"] is True
        assert info["audit_enabled"] is True


class TestMetricsCollectorAbstract:
    """Test the abstract MetricsCollector class."""
    
    def test_cannot_instantiate_abstract_metrics_collector(self):
        """Test that MetricsCollector cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MetricsCollector("test")
    
    def test_metrics_collector_has_required_abstract_methods(self):
        """Test that MetricsCollector has all required abstract methods."""
        abstract_methods = MetricsCollector.__abstractmethods__
        expected_methods = {
            "collect_metric", "get_metric_values", "aggregate_metrics",
            "export_metrics", "configure_retention"
        }
        assert abstract_methods == expected_methods


class TestMetricsCollectorConcrete:
    """Test concrete MetricsCollector implementations."""
    
    def test_metrics_collector_initialization(self, test_metrics_collector):
        """Test MetricsCollector initialization."""
        assert test_metrics_collector.collector_id == "test-collector"
        assert test_metrics_collector.metrics_store == {}
        assert test_metrics_collector.active is True
    
    def test_collect_metric(self, test_metrics_collector):
        """Test metric collection."""
        test_metrics_collector.collect_metric(
            metric_name="response_time",
            value=0.5,
            metric_type=MetricType.TIMER,
            labels={"endpoint": "/api/diagnosis", "method": "POST"}
        )
        
        assert len(test_metrics_collector.collected_metrics) == 1
        metric = test_metrics_collector.collected_metrics[0]
        assert metric["name"] == "response_time"
        assert metric["value"] == 0.5
        assert metric["type"] == MetricType.TIMER
        assert metric["labels"]["endpoint"] == "/api/diagnosis"
    
    def test_get_metric_values(self, test_metrics_collector):
        """Test metric value retrieval."""
        # Collect some metrics
        test_metrics_collector.collect_metric("test_metric", 10, MetricType.COUNTER)
        test_metrics_collector.collect_metric("test_metric", 20, MetricType.COUNTER)
        test_metrics_collector.collect_metric("other_metric", 30, MetricType.COUNTER)
        
        values = test_metrics_collector.get_metric_values("test_metric")
        
        assert len(values) == 2
        assert all(isinstance(v, MetricValue) for v in values)
        assert values[0].value == 10
        assert values[1].value == 20
    
    def test_aggregate_metrics(self, test_metrics_collector):
        """Test metric aggregation."""
        # Collect some metrics
        test_metrics_collector.collect_metric("test_metric", 10, MetricType.COUNTER)
        test_metrics_collector.collect_metric("test_metric", 20, MetricType.COUNTER)
        test_metrics_collector.collect_metric("test_metric", 30, MetricType.COUNTER)
        
        # Test different aggregation types
        sum_result = test_metrics_collector.aggregate_metrics("test_metric", "sum", timedelta(hours=1))
        avg_result = test_metrics_collector.aggregate_metrics("test_metric", "avg", timedelta(hours=1))
        max_result = test_metrics_collector.aggregate_metrics("test_metric", "max", timedelta(hours=1))
        min_result = test_metrics_collector.aggregate_metrics("test_metric", "min", timedelta(hours=1))
        
        assert sum_result == 60
        assert avg_result == 20
        assert max_result == 30
        assert min_result == 10
        
        # Test non-existent metric
        none_result = test_metrics_collector.aggregate_metrics("nonexistent", "sum", timedelta(hours=1))
        assert none_result is None
    
    def test_export_metrics(self, test_metrics_collector):
        """Test metric export."""
        test_metrics_collector.collect_metric("test_metric", 42, MetricType.GAUGE)
        
        exported = test_metrics_collector.export_metrics("prometheus")
        
        assert isinstance(exported, str)
        assert "prometheus" in exported.lower()
        assert "1" in exported  # Should mention the number of metrics
    
    def test_collection_controls(self, test_metrics_collector):
        """Test collection start/stop controls."""
        # Test initial state
        assert test_metrics_collector.is_active() is True
        
        # Test stop
        test_metrics_collector.stop_collection()
        assert test_metrics_collector.is_active() is False
        assert test_metrics_collector.active is False
        
        # Test start
        test_metrics_collector.start_collection()
        assert test_metrics_collector.is_active() is True
        assert test_metrics_collector.active is True
    
    def test_get_collector_info(self, test_metrics_collector):
        """Test collector information retrieval."""
        test_metrics_collector.collect_metric("test", 1, MetricType.COUNTER)
        
        info = test_metrics_collector.get_collector_info()
        
        assert "collector_id" in info
        assert "collector_type" in info
        assert "active" in info
        assert "metrics_count" in info
        
        assert info["collector_id"] == "test-collector"
        assert info["active"] is True


class TestMedicalMetricsCollectorAbstract:
    """Test the abstract MedicalMetricsCollector class."""
    
    def test_cannot_instantiate_abstract_medical_metrics_collector(self):
        """Test that MedicalMetricsCollector cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MedicalMetricsCollector("test")
    
    def test_medical_metrics_collector_has_required_abstract_methods(self):
        """Test that MedicalMetricsCollector has all required abstract methods."""
        abstract_methods = MedicalMetricsCollector.__abstractmethods__
        expected_methods = {
            "collect_metric", "get_metric_values", "aggregate_metrics",
            "export_metrics", "configure_retention", "collect_patient_metric",
            "collect_compliance_metric", "collect_safety_metric", "generate_compliance_report"
        }
        assert abstract_methods == expected_methods


class TestMedicalMetricsCollectorConcrete:
    """Test concrete MedicalMetricsCollector implementations."""
    
    def test_medical_metrics_collector_initialization(self):
        """Test MedicalMetricsCollector initialization."""
        
        class TestMedicalMetricsCollector(MedicalMetricsCollector):
            def collect_metric(self, metric_name, value, metric_type, labels=None):
                pass
            def get_metric_values(self, metric_name, start_time=None, end_time=None):
                return []
            def aggregate_metrics(self, metric_name, aggregation_type, time_window):
                return None
            def export_metrics(self, format="prometheus"):
                return ""
            def configure_retention(self, metric_name, retention_period):
                pass
            def collect_patient_metric(self, patient_id, metric_name, value, anonymize=True):
                pass
            def collect_compliance_metric(self, compliance_metric):
                pass
            def collect_safety_metric(self, safety_event, severity, metadata=None):
                pass
            def generate_compliance_report(self, standard, start_time, end_time):
                return {}
        
        collector = TestMedicalMetricsCollector("test-medical-collector")
        
        assert collector.collector_id == "test-medical-collector"
        assert collector.patient_metrics_enabled is False
        assert collector.compliance_monitoring is True
        assert collector.phi_protection is True
    
    def test_patient_metrics_controls(self):
        """Test patient metrics enable/disable controls."""
        
        class TestMedicalMetricsCollector(MedicalMetricsCollector):
            def collect_metric(self, metric_name, value, metric_type, labels=None):
                pass
            def get_metric_values(self, metric_name, start_time=None, end_time=None):
                return []
            def aggregate_metrics(self, metric_name, aggregation_type, time_window):
                return None
            def export_metrics(self, format="prometheus"):
                return ""
            def configure_retention(self, metric_name, retention_period):
                pass
            def collect_patient_metric(self, patient_id, metric_name, value, anonymize=True):
                pass
            def collect_compliance_metric(self, compliance_metric):
                pass
            def collect_safety_metric(self, safety_event, severity, metadata=None):
                pass
            def generate_compliance_report(self, standard, start_time, end_time):
                return {}
        
        collector = TestMedicalMetricsCollector("test-medical-collector")
        
        # Test enable
        collector.enable_patient_metrics()
        assert collector.patient_metrics_enabled is True
        
        # Test disable
        collector.disable_patient_metrics()
        assert collector.patient_metrics_enabled is False
    
    def test_phi_protection_controls(self):
        """Test PHI protection controls for medical metrics."""
        
        class TestMedicalMetricsCollector(MedicalMetricsCollector):
            def collect_metric(self, metric_name, value, metric_type, labels=None):
                pass
            def get_metric_values(self, metric_name, start_time=None, end_time=None):
                return []
            def aggregate_metrics(self, metric_name, aggregation_type, time_window):
                return None
            def export_metrics(self, format="prometheus"):
                return ""
            def configure_retention(self, metric_name, retention_period):
                pass
            def collect_patient_metric(self, patient_id, metric_name, value, anonymize=True):
                pass
            def collect_compliance_metric(self, compliance_metric):
                pass
            def collect_safety_metric(self, safety_event, severity, metadata=None):
                pass
            def generate_compliance_report(self, standard, start_time, end_time):
                return {}
        
        collector = TestMedicalMetricsCollector("test-medical-collector")
        
        # Test enable
        collector.enable_phi_protection()
        assert collector.phi_protection is True
        
        # Test disable
        collector.disable_phi_protection()
        assert collector.phi_protection is False


class TestCryptoManagerAbstract:
    """Test the abstract CryptoManager class."""
    
    def test_cannot_instantiate_abstract_crypto_manager(self):
        """Test that CryptoManager cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CryptoManager("test")


class TestDataClasses:
    """Test utility data classes."""
    
    def test_log_entry(self):
        """Test LogEntry data class."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            message="Test log message",
            agent_id="agent-001",
            patient_id="patient-123",
            session_id="session-456",
            metadata={"action": "read"},
            phi_detected=False
        )
        
        assert entry.level == "INFO"
        assert entry.message == "Test log message"
        assert entry.agent_id == "agent-001"
        assert entry.patient_id == "patient-123"
        assert entry.session_id == "session-456"
        assert entry.metadata == {"action": "read"}
        assert entry.phi_detected is False
    
    def test_audit_event(self, sample_audit_event):
        """Test AuditEvent data class."""
        assert sample_audit_event.event_type == "data_access"
        assert sample_audit_event.actor == "test_user"
        assert sample_audit_event.resource == "patient_record"
        assert sample_audit_event.action == "read"
        assert sample_audit_event.outcome == "success"
        assert isinstance(sample_audit_event.timestamp, datetime)
    
    def test_metric_value(self):
        """Test MetricValue data class."""
        value = MetricValue(
            value=42.5,
            timestamp=datetime.now(),
            labels={"service": "diagnosis", "version": "1.0"},
            metric_type=MetricType.GAUGE
        )
        
        assert value.value == 42.5
        assert isinstance(value.timestamp, datetime)
        assert value.labels == {"service": "diagnosis", "version": "1.0"}
        assert value.metric_type == MetricType.GAUGE
    
    def test_performance_metric(self):
        """Test PerformanceMetric data class."""
        metric = PerformanceMetric(
            metric_name="response_time",
            value=0.25,
            unit="seconds",
            timestamp=datetime.now(),
            context={"endpoint": "/api/diagnose", "method": "POST"}
        )
        
        assert metric.metric_name == "response_time"
        assert metric.value == 0.25
        assert metric.unit == "seconds"
        assert isinstance(metric.timestamp, datetime)
        assert metric.context == {"endpoint": "/api/diagnose", "method": "POST"}
    
    def test_compliance_metric(self, sample_compliance_metric):
        """Test ComplianceMetric data class."""
        assert sample_compliance_metric.standard == "HIPAA"
        assert sample_compliance_metric.compliance_level == 0.95
        assert sample_compliance_metric.violations_count == 2
        assert isinstance(sample_compliance_metric.timestamp, datetime)
        assert "violations" in sample_compliance_metric.details


class TestMetricType:
    """Test MetricType enum."""
    
    def test_metric_type_values(self):
        """Test MetricType enum values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.TIMER.value == "timer"
        assert MetricType.RATE.value == "rate"
    
    def test_metric_type_membership(self):
        """Test MetricType membership."""
        assert MetricType.COUNTER in MetricType
        assert MetricType.GAUGE in MetricType
        assert MetricType.HISTOGRAM in MetricType
        assert MetricType.TIMER in MetricType
        assert MetricType.RATE in MetricType


class TestAbstractValidators:
    """Test other abstract validator classes."""
    
    def test_cannot_instantiate_abstract_compliance_logger(self):
        """Test that ComplianceLogger cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ComplianceLogger("test", ["HIPAA"])
    
    def test_cannot_instantiate_abstract_phi_detector(self):
        """Test that PHIDetector cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PHIDetector("test")
    
    def test_cannot_instantiate_abstract_log_sanitizer(self):
        """Test that LogSanitizer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LogSanitizer("test")
    
    def test_cannot_instantiate_abstract_secure_logger(self):
        """Test that SecureLogger cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SecureLogger("test")
    
    def test_cannot_instantiate_abstract_performance_monitor(self):
        """Test that PerformanceMonitor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PerformanceMonitor("test")
    
    def test_cannot_instantiate_abstract_compliance_monitor(self):
        """Test that ComplianceMonitor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ComplianceMonitor("test", ["HIPAA"])
    
    def test_cannot_instantiate_abstract_security_metrics_collector(self):
        """Test that SecurityMetricsCollector cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SecurityMetricsCollector("test")


class TestConcreteImplementationHelpers:
    """Test concrete implementation helper methods."""
    
    def test_compliance_logger_abstract_methods(self):
        """Test that ComplianceLogger has required abstract methods."""
        abstract_methods = ComplianceLogger.__abstractmethods__
        expected_methods = {
            "log_compliance_event", "validate_retention_policy",
            "configure_retention_policy", "generate_compliance_report"
        }
        assert abstract_methods == expected_methods
    
    def test_phi_detector_abstract_methods(self):
        """Test that PHIDetector has required abstract methods."""
        abstract_methods = PHIDetector.__abstractmethods__
        expected_methods = {
            "detect_phi", "configure_detection_patterns", "set_sensitivity_level"
        }
        assert abstract_methods == expected_methods
    
    def test_log_sanitizer_abstract_methods(self):
        """Test that LogSanitizer has required abstract methods."""
        abstract_methods = LogSanitizer.__abstractmethods__
        expected_methods = {
            "sanitize_message", "configure_sanitization_rules", "set_masking_strategy"
        }
        assert abstract_methods == expected_methods
    
    def test_secure_logger_abstract_methods(self):
        """Test that SecureLogger has required abstract methods."""
        abstract_methods = SecureLogger.__abstractmethods__
        expected_methods = {
            "log_secure_event", "verify_log_integrity",
            "configure_encryption", "rotate_encryption_keys"
        }
        assert abstract_methods == expected_methods
    
    def test_performance_monitor_abstract_methods(self):
        """Test that PerformanceMonitor has required abstract methods."""
        abstract_methods = PerformanceMonitor.__abstractmethods__
        expected_methods = {
            "monitor_response_time", "monitor_throughput", "monitor_error_rate",
            "check_performance_thresholds", "configure_alert_rules"
        }
        assert abstract_methods == expected_methods
    
    def test_compliance_monitor_abstract_methods(self):
        """Test that ComplianceMonitor has required abstract methods."""
        abstract_methods = ComplianceMonitor.__abstractmethods__
        expected_methods = {
            "monitor_compliance_level", "track_violation",
            "generate_compliance_dashboard", "configure_compliance_alerts"
        }
        assert abstract_methods == expected_methods
    
    def test_security_metrics_collector_abstract_methods(self):
        """Test that SecurityMetricsCollector has required abstract methods."""
        abstract_methods = SecurityMetricsCollector.__abstractmethods__
        expected_methods = {
            "collect_access_metric", "collect_authentication_metric",
            "detect_anomalies", "generate_security_report"
        }
        assert abstract_methods == expected_methods 
"""
Medical metrics collection and monitoring for A2A medical systems.

This module provides abstract metrics frameworks that can be specialized
for different medical domains and monitoring requirements.
"""

from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class MetricValue:
    """Represents a metric value with metadata."""
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: MetricType


@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None


@dataclass
class ComplianceMetric:
    """Represents a compliance metric."""
    standard: str
    compliance_level: float  # 0.0 to 1.0
    violations_count: int
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


class MetricsCollector(ABC):
    """Abstract base class for metrics collection systems.
    
    Provides foundational structure for implementing medical metrics
    collection with compliance and performance monitoring.
    """
    
    def __init__(self, collector_id: str):
        self.collector_id = collector_id
        self.metrics_store: Dict[str, List[MetricValue]] = {}
        self.active = True
    
    @abstractmethod
    def collect_metric(self, 
                      metric_name: str, 
                      value: Union[int, float], 
                      metric_type: MetricType,
                      labels: Optional[Dict[str, str]] = None) -> None:
        """Collect a metric value.
        
        Must be implemented by concrete metrics collectors.
        """
        pass
    
    @abstractmethod
    def get_metric_values(self, 
                         metric_name: str, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[MetricValue]:
        """Get metric values for a specific metric.
        
        Must be implemented by concrete metrics collectors.
        """
        pass
    
    @abstractmethod
    def aggregate_metrics(self, 
                         metric_name: str, 
                         aggregation_type: str,
                         time_window: timedelta) -> Optional[float]:
        """Aggregate metric values over a time window.
        
        Must be implemented by concrete metrics collectors.
        """
        pass
    
    @abstractmethod
    def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in a specific format.
        
        Must be implemented by concrete metrics collectors.
        """
        pass
    
    @abstractmethod
    def configure_retention(self, metric_name: str, retention_period: timedelta) -> None:
        """Configure metric retention policy.
        
        Must be implemented by concrete metrics collectors.
        """
        pass
    
    def start_collection(self) -> None:
        """Start metrics collection."""
        self.active = True
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self.active = False
    
    def is_active(self) -> bool:
        """Check if metrics collection is active."""
        return self.active
    
    def get_collector_info(self) -> Dict[str, Any]:
        """Get information about this metrics collector."""
        return {
            "collector_id": self.collector_id,
            "collector_type": self.__class__.__name__,
            "active": self.active,
            "metrics_count": len(self.metrics_store)
        }


class MedicalMetricsCollector(MetricsCollector):
    """Abstract base class for medical-specific metrics collection.
    
    Extends basic metrics collection with medical domain-specific
    metrics and compliance monitoring.
    """
    
    def __init__(self, collector_id: str):
        super().__init__(collector_id)
        self.patient_metrics_enabled = False
        self.compliance_monitoring = True
        self.phi_protection = True
    
    @abstractmethod
    def collect_patient_metric(self, 
                              patient_id: str, 
                              metric_name: str, 
                              value: Union[int, float],
                              anonymize: bool = True) -> None:
        """Collect a patient-specific metric.
        
        Must be implemented by concrete medical metrics collectors.
        """
        pass
    
    @abstractmethod
    def collect_compliance_metric(self, compliance_metric: ComplianceMetric) -> None:
        """Collect a compliance-related metric.
        
        Must be implemented by concrete medical metrics collectors.
        """
        pass
    
    @abstractmethod
    def collect_safety_metric(self, 
                             safety_event: str, 
                             severity: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """Collect a safety-related metric.
        
        Must be implemented by concrete medical metrics collectors.
        """
        pass
    
    @abstractmethod
    def generate_compliance_report(self, 
                                  standard: str, 
                                  start_time: datetime, 
                                  end_time: datetime) -> Dict[str, Any]:
        """Generate compliance metrics report.
        
        Must be implemented by concrete medical metrics collectors.
        """
        pass
    
    def enable_patient_metrics(self) -> None:
        """Enable patient-specific metrics collection."""
        self.patient_metrics_enabled = True
    
    def disable_patient_metrics(self) -> None:
        """Disable patient-specific metrics collection."""
        self.patient_metrics_enabled = False
    
    def enable_phi_protection(self) -> None:
        """Enable PHI protection in metrics."""
        self.phi_protection = True
    
    def disable_phi_protection(self) -> None:
        """Disable PHI protection in metrics."""
        self.phi_protection = False


class PerformanceMonitor(ABC):
    """Abstract base class for performance monitoring systems.
    
    Provides foundation for implementing performance monitoring
    with medical system-specific considerations.
    """
    
    def __init__(self, monitor_id: str):
        self.monitor_id = monitor_id
        self.performance_thresholds: Dict[str, float] = {}
        self.alerting_enabled = True
    
    @abstractmethod
    def monitor_response_time(self, 
                             operation: str, 
                             response_time: float,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """Monitor response time for an operation.
        
        Must be implemented by concrete performance monitors.
        """
        pass
    
    @abstractmethod
    def monitor_throughput(self, 
                          operation: str, 
                          requests_per_second: float) -> None:
        """Monitor throughput for an operation.
        
        Must be implemented by concrete performance monitors.
        """
        pass
    
    @abstractmethod
    def monitor_error_rate(self, 
                          operation: str, 
                          error_rate: float) -> None:
        """Monitor error rate for an operation.
        
        Must be implemented by concrete performance monitors.
        """
        pass
    
    @abstractmethod
    def check_performance_thresholds(self) -> List[Dict[str, Any]]:
        """Check if performance metrics exceed thresholds.
        
        Must be implemented by concrete performance monitors.
        """
        pass
    
    @abstractmethod
    def configure_alert_rules(self, rules: List[Dict[str, Any]]) -> None:
        """Configure alerting rules for performance monitoring.
        
        Must be implemented by concrete performance monitors.
        """
        pass
    
    def set_performance_threshold(self, metric: str, threshold: float) -> None:
        """Set a performance threshold for a metric."""
        self.performance_thresholds[metric] = threshold
    
    def get_performance_thresholds(self) -> Dict[str, float]:
        """Get all performance thresholds."""
        return self.performance_thresholds.copy()


class ComplianceMonitor(ABC):
    """Abstract base class for compliance monitoring systems.
    
    Provides foundation for implementing compliance monitoring
    against various healthcare standards and regulations.
    """
    
    def __init__(self, monitor_id: str, compliance_standards: List[str]):
        self.monitor_id = monitor_id
        self.compliance_standards = compliance_standards
        self.compliance_thresholds: Dict[str, float] = {}
        self.violation_tracking = True
    
    @abstractmethod
    def monitor_compliance_level(self, 
                               standard: str, 
                               compliance_level: float) -> None:
        """Monitor compliance level for a standard.
        
        Must be implemented by concrete compliance monitors.
        """
        pass
    
    @abstractmethod
    def track_violation(self, 
                       standard: str, 
                       violation_type: str, 
                       severity: str,
                       details: Optional[Dict[str, Any]] = None) -> None:
        """Track a compliance violation.
        
        Must be implemented by concrete compliance monitors.
        """
        pass
    
    @abstractmethod
    def generate_compliance_dashboard(self) -> Dict[str, Any]:
        """Generate compliance dashboard data.
        
        Must be implemented by concrete compliance monitors.
        """
        pass
    
    @abstractmethod
    def configure_compliance_alerts(self, 
                                   standard: str, 
                                   alert_config: Dict[str, Any]) -> None:
        """Configure compliance alerting.
        
        Must be implemented by concrete compliance monitors.
        """
        pass
    
    def add_compliance_standard(self, standard: str) -> None:
        """Add a compliance standard to monitor."""
        if standard not in self.compliance_standards:
            self.compliance_standards.append(standard)
    
    def set_compliance_threshold(self, standard: str, threshold: float) -> None:
        """Set compliance threshold for a standard."""
        self.compliance_thresholds[standard] = threshold
    
    def get_supported_standards(self) -> List[str]:
        """Get list of supported compliance standards."""
        return self.compliance_standards.copy()


class SecurityMetricsCollector(ABC):
    """Abstract base class for security metrics collection.
    
    Provides foundation for implementing security event monitoring
    and threat detection in medical systems.
    """
    
    def __init__(self, collector_id: str):
        self.collector_id = collector_id
        self.threat_detection_enabled = True
        self.security_thresholds: Dict[str, float] = {}
    
    @abstractmethod
    def collect_access_metric(self, 
                             user_id: str, 
                             resource: str, 
                             action: str,
                             success: bool) -> None:
        """Collect access control metrics.
        
        Must be implemented by concrete security metrics collectors.
        """
        pass
    
    @abstractmethod
    def collect_authentication_metric(self, 
                                    user_id: str, 
                                    auth_method: str, 
                                    success: bool) -> None:
        """Collect authentication metrics.
        
        Must be implemented by concrete security metrics collectors.
        """
        pass
    
    @abstractmethod
    def detect_anomalies(self, 
                        time_window: timedelta) -> List[Dict[str, Any]]:
        """Detect security anomalies in metrics.
        
        Must be implemented by concrete security metrics collectors.
        """
        pass
    
    @abstractmethod
    def generate_security_report(self, 
                               start_time: datetime, 
                               end_time: datetime) -> Dict[str, Any]:
        """Generate security metrics report.
        
        Must be implemented by concrete security metrics collectors.
        """
        pass
    
    def enable_threat_detection(self) -> None:
        """Enable threat detection."""
        self.threat_detection_enabled = True
    
    def disable_threat_detection(self) -> None:
        """Disable threat detection."""
        self.threat_detection_enabled = False
    
    def set_security_threshold(self, metric: str, threshold: float) -> None:
        """Set security threshold for a metric."""
        self.security_thresholds[metric] = threshold

"""
Agent discovery protocols for medical A2A systems.

This module provides abstract discovery frameworks that can be specialized
for different medical domains and discovery strategies.
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timedelta

# Import cachetools - assume it's available since we have other dependencies
from cachetools import TTLCache

# A2A SDK imports - confirmed available
from a2a.types import AgentCard


class ComplianceChecker(ABC):
    """Abstract base class for compliance checkers.
    
    Provides foundation for implementing medical compliance validation
    against various healthcare standards.
    """
    
    def __init__(self, checker_id: str, supported_standards: List[str]):
        self.checker_id = checker_id
        self.supported_standards = supported_standards
        self.validation_rules: Dict[str, Any] = {}
    
    @abstractmethod
    async def check_compliance(self, agent: AgentCard, requirements: List[str]) -> bool:
        """Check if an agent meets compliance requirements.
        
        Must be implemented by concrete compliance checkers.
        """
        pass
    
    @abstractmethod
    def configure_compliance_rules(self, standard: str, rules: Dict[str, Any]) -> None:
        """Configure compliance rules for a specific standard.
        
        Must be implemented by concrete compliance checkers.
        """
        pass
    
    @abstractmethod
    def validate_agent_credentials(self, agent: AgentCard) -> bool:
        """Validate agent credentials against compliance standards.
        
        Must be implemented by concrete compliance checkers.
        """
        pass
    
    def get_supported_standards(self) -> List[str]:
        """Get list of supported compliance standards."""
        return self.supported_standards.copy()
    
    def add_standard(self, standard: str) -> None:
        """Add a compliance standard."""
        if standard not in self.supported_standards:
            self.supported_standards.append(standard)


class AgentRegistry(ABC):
    """Abstract base class for agent registries.
    
    Provides foundation for implementing agent registration and
    discovery systems with different storage backends.
    """
    
    def __init__(self, registry_id: str):
        self.registry_id = registry_id
    
    @abstractmethod
    async def register_agent(self, agent: AgentCard, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register an agent in the registry.
        
        Must be implemented by concrete registries.
        """
        pass
    
    @abstractmethod
    async def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an agent from the registry.
        
        Must be implemented by concrete registries.
        """
        pass
    
    @abstractmethod
    async def query_agents(self, agent_type: Optional[str] = None, capabilities: Optional[List[str]] = None) -> List[AgentCard]:
        """Query agents by type and capabilities.
        
        Must be implemented by concrete registries.
        """
        pass
    
    @abstractmethod
    async def get_agent(self, agent_name: str) -> Optional[AgentCard]:
        """Get a specific agent by name.
        
        Must be implemented by concrete registries.
        """
        pass
    
    @abstractmethod
    async def list_agents(self) -> List[AgentCard]:
        """List all registered agents.
        
        Must be implemented by concrete registries.
        """
        pass
    
    @abstractmethod
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Must be implemented by concrete registries.
        """
        pass


class AgentDiscovery(ABC):
    """Abstract base class for agent discovery services.
    
    Provides foundation for implementing agent discovery with
    different search strategies and compliance requirements.
    """
    
    def __init__(self, discovery_id: str, registry: AgentRegistry, compliance_checker: Optional[ComplianceChecker] = None):
        self.discovery_id = discovery_id
        self.registry = registry
        self.compliance_checker = compliance_checker
        self._cache = TTLCache(maxsize=100, ttl=300)  # 5-minute cache
    
    @abstractmethod
    async def discover_agent(
        self,
        agent_type: str,
        required_capabilities: List[str],
        compliance_requirements: Optional[List[str]] = None
    ) -> Optional[AgentCard]:
        """Discover a single agent matching criteria.
        
        Must be implemented by concrete discovery services.
        """
        pass
    
    @abstractmethod
    async def discover_multiple_agents(
        self,
        agent_type: str,
        required_capabilities: List[str],
        compliance_requirements: Optional[List[str]] = None,
        max_agents: int = 5
    ) -> List[AgentCard]:
        """Discover multiple agents matching criteria.
        
        Must be implemented by concrete discovery services.
        """
        pass
    
    @abstractmethod
    def configure_discovery_strategy(self, strategy: Dict[str, Any]) -> None:
        """Configure the discovery strategy.
        
        Must be implemented by concrete discovery services.
        """
        pass
    
    async def _verify_compliance(
        self,
        agent: AgentCard,
        requirements: List[str]
    ) -> bool:
        """Verify agent meets compliance requirements."""
        if not self.compliance_checker:
            return True
        return await self.compliance_checker.check_compliance(agent, requirements)
    
    def _select_best_agent(self, agents: List[AgentCard]) -> Optional[AgentCard]:
        """Select the best agent from a list of candidates.
        
        Default implementation returns the first agent.
        Can be overridden by concrete implementations.
        """
        return agents[0] if agents else None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_maxsize": getattr(self._cache, 'maxsize', 0),
            "cache_ttl": getattr(self._cache, 'ttl', 0)
        }
    
    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._cache.clear()


class MedicalAgentDiscovery(AgentDiscovery):
    """Abstract base class for medical agent discovery services.
    
    Extends basic agent discovery with medical-specific functionality
    and compliance requirements.
    """
    
    def __init__(self, discovery_id: str, registry: AgentRegistry, compliance_checker: Optional[ComplianceChecker] = None):
        super().__init__(discovery_id, registry, compliance_checker)
        self.medical_specialties: List[str] = []
        self.emergency_capabilities: List[str] = []
    
    @abstractmethod
    async def discover_by_medical_specialty(
        self,
        specialty: str,
        compliance_requirements: Optional[List[str]] = None
    ) -> List[AgentCard]:
        """Discover agents by medical specialty.
        
        Must be implemented by concrete medical discovery services.
        """
        pass
    
    @abstractmethod
    async def discover_emergency_agents(
        self,
        urgency_level: int = 5
    ) -> List[AgentCard]:
        """Discover agents capable of handling medical emergencies.
        
        Must be implemented by concrete medical discovery services.
        """
        pass
    
    @abstractmethod
    def configure_medical_criteria(self, criteria: Dict[str, Any]) -> None:
        """Configure medical-specific discovery criteria.
        
        Must be implemented by concrete medical discovery services.
        """
        pass
    
    def add_medical_specialty(self, specialty: str) -> None:
        """Add a medical specialty to the discovery service."""
        if specialty not in self.medical_specialties:
            self.medical_specialties.append(specialty)
    
    def add_emergency_capability(self, capability: str) -> None:
        """Add an emergency capability to the discovery service."""
        if capability not in self.emergency_capabilities:
            self.emergency_capabilities.append(capability)
    
    def get_supported_specialties(self) -> List[str]:
        """Get list of supported medical specialties."""
        return self.medical_specialties.copy()
    
    def get_emergency_capabilities(self) -> List[str]:
        """Get list of emergency capabilities."""
        return self.emergency_capabilities.copy()


class SpecialtyAgentDiscovery(MedicalAgentDiscovery):
    """Abstract base class for specialty-specific agent discovery.
    
    Provides foundation for implementing discovery services that
    focus on specific medical specialties or domains.
    """
    
    def __init__(self, discovery_id: str, specialty: str, registry: AgentRegistry, compliance_checker: Optional[ComplianceChecker] = None):
        super().__init__(discovery_id, registry, compliance_checker)
        self.primary_specialty = specialty
        self.subspecialties: List[str] = []
        self.specialty_requirements: Dict[str, Any] = {}
    
    @abstractmethod
    async def discover_specialists(
        self,
        subspecialty: Optional[str] = None,
        experience_level: Optional[str] = None
    ) -> List[AgentCard]:
        """Discover specialist agents in this specialty.
        
        Must be implemented by concrete specialty discovery services.
        """
        pass
    
    @abstractmethod
    def configure_specialty_requirements(self, requirements: Dict[str, Any]) -> None:
        """Configure requirements specific to this specialty.
        
        Must be implemented by concrete specialty discovery services.
        """
        pass
    
    def add_subspecialty(self, subspecialty: str) -> None:
        """Add a subspecialty to this discovery service."""
        if subspecialty not in self.subspecialties:
            self.subspecialties.append(subspecialty)


class GeographicAgentDiscovery(AgentDiscovery):
    """Abstract base class for geographic agent discovery.
    
    Provides foundation for implementing location-based agent
    discovery with geographic constraints and preferences.
    """
    
    def __init__(self, discovery_id: str, registry: AgentRegistry, compliance_checker: Optional[ComplianceChecker] = None):
        super().__init__(discovery_id, registry, compliance_checker)
        self.geographic_regions: List[str] = []
        self.distance_preferences: Dict[str, float] = {}
    
    @abstractmethod
    async def discover_by_location(
        self,
        location: Dict[str, Any],
        radius: float,
        agent_type: str
    ) -> List[AgentCard]:
        """Discover agents within a geographic area.
        
        Must be implemented by concrete geographic discovery services.
        """
        pass
    
    @abstractmethod
    def configure_geographic_constraints(self, constraints: Dict[str, Any]) -> None:
        """Configure geographic discovery constraints.
        
        Must be implemented by concrete geographic discovery services.
        """
        pass
    
    def add_geographic_region(self, region: str) -> None:
        """Add a geographic region to the discovery service."""
        if region not in self.geographic_regions:
            self.geographic_regions.append(region)


# Backward compatibility aliases
HealthcareAgentDiscovery = MedicalAgentDiscovery
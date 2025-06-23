"""
OMOP CDM (Common Data Model) models for medical A2A systems.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class VocabularyType(str, Enum):
    """OMOP vocabulary types."""
    SNOMED_CT = "SNOMED"
    ICD_10_CM = "ICD10CM"
    ICD_10_PCS = "ICD10PCS"
    LOINC = "LOINC"
    RXNORM = "RxNorm"
    CPT = "CPT"
    HCPCS = "HCPCS"
    NDC = "NDC"
    GENDER = "Gender"
    RACE = "Race"
    ETHNICITY = "Ethnicity"


class DomainType(str, Enum):
    """OMOP domain types."""
    CONDITION = "Condition"
    DRUG = "Drug"
    PROCEDURE = "Procedure"
    OBSERVATION = "Observation"
    MEASUREMENT = "Measurement"
    VISIT = "Visit"
    DEVICE = "Device"
    SPECIMEN = "Specimen"
    NOTE = "Note"
    COST = "Cost"


class ConceptClassType(str, Enum):
    """OMOP concept class types."""
    CLINICAL_FINDING = "Clinical Finding"
    PROCEDURE = "Procedure"
    DRUG = "Drug"
    MEASUREMENT = "Measurement"
    OBSERVATION = "Observation"
    DEVICE = "Device"
    SPECIMEN = "Specimen"
    VISIT = "Visit"
    ORGANISM = "Organism"
    PHARMA_SUBSTANCE = "Pharma/Biol Product"


class OMOPVocabulary(BaseModel):
    """OMOP vocabulary model."""
    vocabulary_id: str = Field(..., description="Vocabulary identifier")
    vocabulary_name: str = Field(..., description="Vocabulary name")
    vocabulary_reference: Optional[str] = Field(None, description="Reference URL or documentation")
    vocabulary_version: Optional[str] = Field(None, description="Vocabulary version")
    vocabulary_concept_id: Optional[int] = Field(None, description="Concept ID for the vocabulary")
    is_active: bool = Field(True, description="Whether the vocabulary is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OMOPConcept(BaseModel):
    """OMOP concept model."""
    concept_id: int = Field(..., description="Unique concept identifier")
    concept_name: str = Field(..., description="Human-readable concept name")
    domain_id: DomainType = Field(..., description="Concept domain")
    vocabulary_id: VocabularyType = Field(..., description="Source vocabulary")
    concept_class_id: ConceptClassType = Field(..., description="Concept class")
    standard_concept: Optional[str] = Field(None, description="Standard concept flag")
    concept_code: str = Field(..., description="Original code in source vocabulary")
    valid_start_date: datetime = Field(..., description="Start date of concept validity")
    valid_end_date: datetime = Field(..., description="End date of concept validity")
    invalid_reason: Optional[str] = Field(None, description="Reason for invalidation")
    synonyms: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('valid_end_date')
    def validate_dates(cls, v, values):
        """Validate that end date is after start date."""
        if 'valid_start_date' in values and v <= values['valid_start_date']:
            raise ValueError("Valid end date must be after valid start date")
        return v
    
    def is_valid(self, reference_date: Optional[datetime] = None) -> bool:
        """Check if the concept is valid at the given date."""
        if reference_date is None:
            reference_date = datetime.utcnow()
        
        return (self.valid_start_date <= reference_date <= self.valid_end_date and 
                self.invalid_reason is None)
    
    def add_synonym(self, synonym: str) -> None:
        """Add a synonym to the concept."""
        if synonym not in self.synonyms:
            self.synonyms.append(synonym)
    
    def get_domain(self) -> str:
        """Get the domain as a string."""
        return self.domain_id.value
    
    def get_vocabulary(self) -> str:
        """Get the vocabulary as a string."""
        return self.vocabulary_id.value


class OMOPConceptRelationship(BaseModel):
    """OMOP concept relationship model."""
    concept_id_1: int = Field(..., description="First concept ID")
    concept_id_2: int = Field(..., description="Second concept ID")
    relationship_id: str = Field(..., description="Relationship type")
    valid_start_date: datetime = Field(..., description="Start date of relationship validity")
    valid_end_date: datetime = Field(..., description="End date of relationship validity")
    invalid_reason: Optional[str] = Field(None, description="Reason for invalidation")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OMOPConceptAncestor(BaseModel):
    """OMOP concept ancestor model."""
    ancestor_concept_id: int = Field(..., description="Ancestor concept ID")
    descendant_concept_id: int = Field(..., description="Descendant concept ID")
    min_levels_of_separation: int = Field(..., description="Minimum levels of separation")
    max_levels_of_separation: int = Field(..., description="Maximum levels of separation")
    
    @validator('min_levels_of_separation')
    def validate_min_levels(cls, v):
        """Validate minimum levels of separation."""
        if v < 0:
            raise ValueError("Minimum levels of separation must be non-negative")
        return v
    
    @validator('max_levels_of_separation')
    def validate_max_levels(cls, v, values):
        """Validate maximum levels of separation."""
        if 'min_levels_of_separation' in values and v < values['min_levels_of_separation']:
            raise ValueError("Maximum levels must be >= minimum levels")
        return v


class OMOPConceptSynonym(BaseModel):
    """OMOP concept synonym model."""
    concept_id: int = Field(..., description="Concept ID")
    concept_synonym_name: str = Field(..., description="Synonym name")
    language_concept_id: Optional[int] = Field(None, description="Language concept ID")


class OMOPConceptService:
    """Service for working with OMOP concepts."""
    
    def __init__(self):
        self.concepts: Dict[int, OMOPConcept] = {}
        self.vocabularies: Dict[str, OMOPVocabulary] = {}
        self.relationships: List[OMOPConceptRelationship] = []
        self.ancestors: List[OMOPConceptAncestor] = []
    
    def add_concept(self, concept: OMOPConcept) -> None:
        """Add a concept to the service."""
        self.concepts[concept.concept_id] = concept
    
    def get_concept(self, concept_id: int) -> Optional[OMOPConcept]:
        """Get a concept by ID."""
        return self.concepts.get(concept_id)
    
    def search_concepts(self, query: str, domain: Optional[DomainType] = None, 
                       vocabulary: Optional[VocabularyType] = None) -> List[OMOPConcept]:
        """Search for concepts by name."""
        results = []
        query_lower = query.lower()
        
        for concept in self.concepts.values():
            # Check if concept matches search criteria
            if query_lower in concept.concept_name.lower():
                if domain and concept.domain_id != domain:
                    continue
                if vocabulary and concept.vocabulary_id != vocabulary:
                    continue
                results.append(concept)
            
            # Also check synonyms
            for synonym in concept.synonyms:
                if query_lower in synonym.lower():
                    if domain and concept.domain_id != domain:
                        continue
                    if vocabulary and concept.vocabulary_id != vocabulary:
                        continue
                    results.append(concept)
                    break
        
        return results
    
    def get_concept_ancestors(self, concept_id: int, max_levels: Optional[int] = None) -> List[OMOPConcept]:
        """Get ancestor concepts."""
        ancestor_ids = set()
        
        for ancestor in self.ancestors:
            if ancestor.descendant_concept_id == concept_id:
                if max_levels is None or ancestor.max_levels_of_separation <= max_levels:
                    ancestor_ids.add(ancestor.ancestor_concept_id)
        
        return [concept for concept_id in ancestor_ids 
                if (concept := self.concepts.get(concept_id)) is not None]
    
    def get_concept_descendants(self, concept_id: int, max_levels: Optional[int] = None) -> List[OMOPConcept]:
        """Get descendant concepts."""
        descendant_ids = set()
        
        for ancestor in self.ancestors:
            if ancestor.ancestor_concept_id == concept_id:
                if max_levels is None or ancestor.max_levels_of_separation <= max_levels:
                    descendant_ids.add(ancestor.descendant_concept_id)
        
        return [concept for concept_id in descendant_ids 
                if (concept := self.concepts.get(concept_id)) is not None]
    
    def get_concept_relationships(self, concept_id: int, relationship_type: Optional[str] = None) -> List[OMOPConceptRelationship]:
        """Get relationships for a concept."""
        results = []
        
        for relationship in self.relationships:
            if (relationship.concept_id_1 == concept_id or relationship.concept_id_2 == concept_id):
                if relationship_type is None or relationship.relationship_id == relationship_type:
                    results.append(relationship)
        
        return results
    
    def add_vocabulary(self, vocabulary: OMOPVocabulary) -> None:
        """Add a vocabulary to the service."""
        self.vocabularies[vocabulary.vocabulary_id] = vocabulary
    
    def get_vocabulary(self, vocabulary_id: str) -> Optional[OMOPVocabulary]:
        """Get a vocabulary by ID."""
        return self.vocabularies.get(vocabulary_id)
    
    def get_concepts_by_vocabulary(self, vocabulary_id: VocabularyType) -> List[OMOPConcept]:
        """Get all concepts from a specific vocabulary."""
        return [concept for concept in self.concepts.values() 
                if concept.vocabulary_id == vocabulary_id]
    
    def get_concepts_by_domain(self, domain: DomainType) -> List[OMOPConcept]:
        """Get all concepts from a specific domain."""
        return [concept for concept in self.concepts.values() 
                if concept.domain_id == domain]

class OMOPQuery(BaseModel):
    """Query structure for OMOP CDM databases."""
    target_domains: List[str]
    concept_ids: List[int]
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range with start and end dates")
    patient_cohort: Optional[str]
    aggregation_type: Optional[str]
    
class OMOPResult(BaseModel):
    """Result from OMOP CDM query."""
    query_id: str
    row_count: int
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    execution_time_ms: float
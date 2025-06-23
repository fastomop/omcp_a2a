from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MedicalQueryType(str, Enum):
    """Types of medical queries the system can handle."""
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    MEDICATION = "medication"
    LAB_RESULT = "lab_result"
    PATIENT_HISTORY = "patient_history"
    CLINICAL_GUIDELINE = "clinical_guideline"

class MedicalConcept(BaseModel):
    """Represents a medical concept with semantic information."""
    concept_id: str = Field(..., description="Unique concept identifier (e.g., OMOP concept ID)")
    concept_name: str = Field(..., description="Human-readable concept name")
    domain: str = Field(..., description="Medical domain (Drug, Condition, Procedure)")
    vocabulary_id: str = Field(..., description="Source vocabulary (SNOMED, RxNorm, etc.)")
    concept_code: str = Field(..., description="Original code in source vocabulary")
    synonyms: List[str] = Field(default_factory=list)
    
    @validator('domain')
    def validate_domain(cls, v):
        valid_domains = ['Drug', 'Condition', 'Procedure', 'Observation', 'Measurement']
        if v not in valid_domains:
            raise ValueError(f"Domain must be one of {valid_domains}")
        return v

class EvidenceSource(BaseModel):
    """Represents a source of medical evidence."""
    source_id: str = Field(..., description="Unique source identifier")
    source_type: str = Field(..., description="Type of source (clinical_trial, guideline, case_study, etc.)")
    title: str = Field(..., description="Title or name of the source")
    authors: List[str] = Field(default_factory=list)
    publication_date: Optional[datetime] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance to the query")
    evidence_strength: str = Field("moderate", description="Strength of evidence (weak, moderate, strong)")

class PatientContext(BaseModel):
    """Patient context for medical queries."""
    patient_id: Optional[str] = Field(None, description="De-identified patient ID")
    age_group: Optional[str] = Field(None, description="Age group (e.g., 18-25, 65+)")
    relevant_conditions: List[MedicalConcept] = Field(default_factory=list)
    current_medications: List[MedicalConcept] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    
class MedicalQuery(BaseModel):
    """Structured medical query."""
    query_id: str = Field(..., description="Unique query identifier")
    query_type: MedicalQueryType
    query_text: str = Field(..., description="Original query text")
    extracted_concepts: List[MedicalConcept] = Field(default_factory=list)
    patient_context: Optional[PatientContext] = None
    temporal_context: Optional[str] = Field(None, description="Time context (current, past 30 days, etc.)")
    urgency_level: int = Field(1, ge=1, le=5, description="1=routine, 5=urgent")
    
class MedicalAnswer(BaseModel):
    """Structured medical answer with evidence."""
    answer_id: str
    query_id: str
    answer_text: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    evidence_sources: List[EvidenceSource] = Field(default_factory=list)
    relevant_guidelines: List[str] = Field(default_factory=list)
    disclaimers: List[str] = Field(default_factory=list)
    requires_human_review: bool = False

class Patient(BaseModel):
    """Patient information model."""
    patient_id: str = Field(..., description="Unique patient identifier")
    mrn: Optional[str] = Field(None, description="Medical record number")
    first_name: Optional[str] = Field(None, description="Patient first name")
    last_name: Optional[str] = Field(None, description="Patient last name")
    date_of_birth: Optional[datetime] = Field(None, description="Patient date of birth")
    gender: Optional[str] = Field(None, description="Patient gender")
    contact_info: Dict[str, str] = Field(default_factory=dict)
    insurance_info: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Provider(BaseModel):
    """Healthcare provider information model."""
    provider_id: str = Field(..., description="Unique provider identifier")
    npi: Optional[str] = Field(None, description="National Provider Identifier")
    first_name: str = Field(..., description="Provider first name")
    last_name: str = Field(..., description="Provider last name")
    specialty: Optional[str] = Field(None, description="Medical specialty")
    credentials: List[str] = Field(default_factory=list)
    organization_id: Optional[str] = Field(None, description="Associated organization")
    contact_info: Dict[str, str] = Field(default_factory=dict)
    is_active: bool = Field(True, description="Whether the provider is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MedicalRecord(BaseModel):
    """Medical record model."""
    record_id: str = Field(..., description="Unique record identifier")
    patient_id: str = Field(..., description="Associated patient ID")
    provider_id: Optional[str] = Field(None, description="Associated provider ID")
    record_type: str = Field(..., description="Type of medical record")
    record_date: datetime = Field(..., description="Date of the medical record")
    content: Dict[str, Any] = Field(..., description="Record content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_encrypted: bool = Field(False, description="Whether the record is encrypted")
    access_level: str = Field("standard", description="Access level required")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
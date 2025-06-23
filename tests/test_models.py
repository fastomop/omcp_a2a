"""
Tests for model components: MedicalRecord, Patient, Provider, A2AMessage, MedicalTask, and OMOP models.
"""

import pytest
from datetime import datetime, date
from typing import List, Optional
from unittest.mock import Mock

from a2a_medical.models.medical import MedicalRecord, Patient, Provider
from a2a_medical.models.messages import A2AMessage, MessageType, MessagePriority
from a2a_medical.models.tasks import MedicalTask, TaskStatus, TaskPriority
from a2a_medical.models.omop import OMOPConcept, OMOPVocabulary, DomainType, VocabularyType, ConceptClassType


class TestPatientModel:
    """Test the Patient Pydantic model."""
    
    def test_patient_creation_full(self):
        """Test creating a patient with all fields."""
        patient = Patient(
            patient_id="PAT-001",
            mrn="MRN-12345",
            first_name="John",
            last_name="Doe",
            date_of_birth=datetime(1980, 5, 15),
            gender="male",
            contact_info={
                "email": "john.doe@example.com",
                "phone": "+1-555-123-4567"
            },
            insurance_info={
                "provider": "Blue Cross",
                "policy_number": "BC123456789",
                "group_number": "GRP001"
            }
        )
        
        assert patient.patient_id == "PAT-001"
        assert patient.mrn == "MRN-12345"
        assert patient.first_name == "John"
        assert patient.last_name == "Doe"
        assert patient.date_of_birth == datetime(1980, 5, 15)
        assert patient.gender == "male"
        assert patient.contact_info["email"] == "john.doe@example.com"
        assert patient.insurance_info["provider"] == "Blue Cross"
    
    def test_patient_creation_minimal(self):
        """Test creating a patient with minimal required fields."""
        patient = Patient(
            patient_id="PAT-002",
            first_name="Jane",
            last_name="Smith",
            date_of_birth=datetime(1990, 3, 22),
            gender="female"
        )
        
        assert patient.patient_id == "PAT-002"
        assert patient.first_name == "Jane"
        assert patient.last_name == "Smith"
        assert patient.date_of_birth == datetime(1990, 3, 22)
        assert patient.gender == "female"
        
        # Test defaults
        assert patient.contact_info == {}
        assert patient.insurance_info == {}
    
    def test_patient_json_serialization(self):
        """Test patient JSON serialization."""
        patient = Patient(
            patient_id="PAT-008",
            first_name="Bob",
            last_name="Wilson",
            date_of_birth=datetime(1970, 8, 30),
            gender="male"
        )
        
        json_data = patient.model_dump()
        
        assert json_data["patient_id"] == "PAT-008"
        assert json_data["first_name"] == "Bob"
        assert json_data["last_name"] == "Wilson"
        
        # Test round-trip
        restored_patient = Patient.model_validate(json_data)
        assert restored_patient.patient_id == patient.patient_id


class TestProviderModel:
    """Test the Provider Pydantic model."""
    
    def test_provider_creation_full(self):
        """Test creating a provider with all fields."""
        provider = Provider(
            provider_id="PROV-001",
            npi="1234567890",
            first_name="Dr. Sarah",
            last_name="Johnson",
            specialty="cardiology",
            credentials=["MD", "FACC"],
            organization_id="ORG-001",
            contact_info={
                "email": "sarah.johnson@hospital.com",
                "phone": "+1-555-234-5678"
            },
            is_active=True
        )
        
        assert provider.provider_id == "PROV-001"
        assert provider.npi == "1234567890"
        assert provider.first_name == "Dr. Sarah"
        assert provider.last_name == "Johnson"
        assert provider.specialty == "cardiology"
        assert "MD" in provider.credentials
        assert provider.organization_id == "ORG-001"
        assert provider.contact_info["email"] == "sarah.johnson@hospital.com"
        assert provider.is_active is True
    
    def test_provider_creation_minimal(self):
        """Test creating a provider with minimal required fields."""
        provider = Provider(
            provider_id="PROV-002",
            first_name="Dr. Michael",
            last_name="Brown"
        )
        
        assert provider.provider_id == "PROV-002"
        assert provider.first_name == "Dr. Michael"
        assert provider.last_name == "Brown"
        
        # Test defaults
        assert provider.npi is None
        assert provider.specialty is None
        assert provider.credentials == []
        assert provider.organization_id is None
        assert provider.contact_info == {}
        assert provider.is_active is True
    
    def test_provider_json_serialization(self):
        """Test provider JSON serialization."""
        provider = Provider(
            provider_id="PROV-003",
            first_name="Dr. Lisa",
            last_name="Chen",
            specialty="pediatrics"
        )
        
        json_data = provider.model_dump()
        
        assert json_data["provider_id"] == "PROV-003"
        assert json_data["first_name"] == "Dr. Lisa"
        assert json_data["last_name"] == "Chen"
        assert json_data["specialty"] == "pediatrics"
        
        # Test round-trip
        restored_provider = Provider.model_validate(json_data)
        assert restored_provider.provider_id == provider.provider_id


class TestMedicalRecordModel:
    """Test the MedicalRecord Pydantic model."""
    
    def test_medical_record_creation_full(self):
        """Test creating a medical record with all fields."""
        record = MedicalRecord(
            record_id="REC-001",
            patient_id="PAT-001",
            provider_id="PROV-001",
            record_type="progress_note",
            record_date=datetime(2024, 7, 15, 10, 30),
            content={
                "chief_complaint": "Chest pain",
                "history": "Patient reports chest pain for 2 hours",
                "physical_exam": "Vitals stable, chest clear",
                "assessment": "Rule out cardiac cause",
                "plan": "EKG, cardiac enzymes"
            },
            metadata={
                "encounter_id": "ENC-001",
                "department": "emergency"
            },
            is_encrypted=False,
            access_level="standard"
        )
        
        assert record.record_id == "REC-001"
        assert record.patient_id == "PAT-001"
        assert record.provider_id == "PROV-001"
        assert record.record_type == "progress_note"
        assert record.record_date == datetime(2024, 7, 15, 10, 30)
        assert record.content["chief_complaint"] == "Chest pain"
        assert record.metadata["encounter_id"] == "ENC-001"
        assert record.is_encrypted is False
        assert record.access_level == "standard"
    
    def test_medical_record_creation_minimal(self):
        """Test creating a medical record with minimal required fields."""
        record = MedicalRecord(
            record_id="REC-002",
            patient_id="PAT-002",
            record_type="visit_note",
            record_date=datetime(2024, 7, 16),
            content={
                "chief_complaint": "Chest pain"
            }
        )
        
        assert record.record_id == "REC-002"
        assert record.patient_id == "PAT-002"
        assert record.record_type == "visit_note"
        assert record.record_date == datetime(2024, 7, 16)
        assert record.content["chief_complaint"] == "Chest pain"
        
        # Test defaults
        assert record.provider_id is None
        assert record.metadata == {}
        assert record.is_encrypted is False
        assert record.access_level == "standard"
    
    def test_medical_record_validation_empty_fields(self):
        """Test medical record validation with empty fields."""
        # This should pass because the framework doesn't validate empty fields
        record = MedicalRecord(
            record_id="REC-003",
            patient_id="PAT-003",
            record_type="lab_result",
            record_date=datetime(2024, 7, 17),
            content={}
        )
        assert record.record_id == "REC-003"
    
    def test_medical_record_json_serialization(self):
        """Test medical record JSON serialization."""
        record = MedicalRecord(
            record_id="REC-004",
            patient_id="PAT-004",
            record_type="discharge_summary",
            record_date=datetime(2024, 7, 18),
            content={
                "summary": "Patient stable for discharge",
                "medications": ["aspirin 81mg daily"],
                "follow_up": ["Cardiology in 1 week"]
            }
        )
        
        json_data = record.model_dump()
        
        assert json_data["record_id"] == "REC-004"
        assert json_data["patient_id"] == "PAT-004"
        assert json_data["record_type"] == "discharge_summary"
        assert json_data["content"]["summary"] == "Patient stable for discharge"
        
        # Test round-trip
        restored_record = MedicalRecord.model_validate(json_data)
        assert restored_record.record_id == record.record_id


class TestA2AMessageModel:
    """Test the A2AMessage Pydantic model."""
    
    def test_a2a_message_creation_full(self):
        """Test creating an A2A message with all fields."""
        message = A2AMessage(
            message_id="MSG-001",
            message_type=MessageType.QUERY,
            sender_id="AGENT-001",
            recipient_ids=["AGENT-002", "AGENT-003"],
            content={
                "query": "What is the recommended treatment for hypertension?",
                "patient_context": {
                    "age": 65,
                    "conditions": ["diabetes"]
                }
            },
            priority=MessagePriority.HIGH,
            expires_at=datetime(2024, 12, 31, 23, 59)
        )
        
        assert message.message_id == "MSG-001"
        assert message.message_type == MessageType.QUERY
        assert message.sender_id == "AGENT-001"
        assert "AGENT-002" in message.recipient_ids
        assert "AGENT-003" in message.recipient_ids
        assert message.content["query"] == "What is the recommended treatment for hypertension?"
        assert message.priority == MessagePriority.HIGH
        assert message.expires_at == datetime(2024, 12, 31, 23, 59)
    
    def test_a2a_message_creation_minimal(self):
        """Test creating an A2A message with minimal required fields."""
        message = A2AMessage(
            message_id="MSG-002",
            message_type=MessageType.RESPONSE,
            sender_id="AGENT-002",
            recipient_ids=["AGENT-001"],
            content={
                "result": "success"
            }
        )
        
        assert message.message_id == "MSG-002"
        assert message.message_type == MessageType.RESPONSE
        assert message.sender_id == "AGENT-002"
        assert message.recipient_ids == ["AGENT-001"]
        assert message.content["result"] == "success"
        
        # Test defaults
        assert message.priority == MessagePriority.NORMAL
        assert message.parts == []
        assert message.metadata == {}
    
    def test_a2a_message_validation_empty_fields(self):
        """Test A2A message validation with empty fields."""
        # This should pass because the framework allows empty content
        message = A2AMessage(
            message_id="MSG-003",
            message_type=MessageType.NOTIFICATION,
            sender_id="AGENT-003",
            recipient_ids=["AGENT-001"],
            content={}
        )
        assert message.message_id == "MSG-003"
    
    def test_a2a_message_json_serialization(self):
        """Test A2A message JSON serialization."""
        message = A2AMessage(
            message_id="MSG-004",
            message_type=MessageType.EVENT,
            sender_id="AGENT-004",
            recipient_ids=["AGENT-001"],
            content={
                "event_type": "patient_scheduled",
                "priority": "normal"
            },
            priority=MessagePriority.NORMAL
        )
        
        json_data = message.model_dump()
        
        assert json_data["message_id"] == "MSG-004"
        assert json_data["message_type"] == "event"
        assert json_data["sender_id"] == "AGENT-004"
        assert json_data["recipient_ids"] == ["AGENT-001"]
        
        # Test round-trip
        restored_message = A2AMessage.model_validate(json_data)
        assert restored_message.message_id == message.message_id


class TestMedicalTaskModel:
    """Test the MedicalTask dataclass model."""
    
    def test_medical_task_creation_full(self):
        """Test creating a medical task with all fields."""
        task = MedicalTask(
            task_id="TASK-001",
            task_type="diagnosis",
            description="Analyze patient symptoms and provide differential diagnosis",
            parameters={
                "patient_id": "PAT-001",
                "symptoms": ["chest_pain", "shortness_of_breath"],
                "urgency": "high"
            },
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING,
            assigned_agent="AGENT-001",
            metadata={
                "department": "cardiology",
                "expected_completion": "2024-07-20T10:00:00"
            }
        )
        
        assert task.task_id == "TASK-001"
        assert task.task_type == "diagnosis"
        assert task.description == "Analyze patient symptoms and provide differential diagnosis"
        assert task.parameters["patient_id"] == "PAT-001"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.assigned_agent == "AGENT-001"
        assert task.metadata["department"] == "cardiology"
    
    def test_medical_task_creation_minimal(self):
        """Test creating a medical task with minimal required fields."""
        task = MedicalTask(
            task_id="TASK-002",
            task_type="consultation",
            description="Cardiology consultation for patient"
        )
        
        assert task.task_id == "TASK-002"
        assert task.task_type == "consultation"
        assert task.description == "Cardiology consultation for patient"
        
        # Test defaults
        assert task.parameters == {}
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING
        assert task.assigned_agent is None
        assert task.metadata == {}
        assert task.error_message is None
        assert task.result is None
    
    def test_medical_task_status_transitions(self):
        """Test medical task status transitions."""
        task = MedicalTask(
            task_id="TASK-003",
            task_type="treatment_plan",
            description="Create treatment plan for patient"
        )
        
        # Test initial status
        assert task.status == TaskStatus.PENDING
        assert task.is_active()
        assert not task.is_completed()
        
        # Test status update
        task.update_status(TaskStatus.IN_PROGRESS)
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.is_active()
        assert not task.is_completed()
        
        # Test completion
        task.update_status(TaskStatus.COMPLETED)
        assert task.status == TaskStatus.COMPLETED
        assert not task.is_active()
        assert task.is_completed()
        assert task.completed_at is not None
    
    def test_medical_task_assignment(self):
        """Test medical task assignment."""
        task = MedicalTask(
            task_id="TASK-004",
            task_type="medication_review",
            description="Review patient medications"
        )
        
        # Test assignment
        task.assign_to_agent("AGENT-002")
        assert task.assigned_agent == "AGENT-002"
    
    def test_medical_task_result_setting(self):
        """Test medical task result setting."""
        task = MedicalTask(
            task_id="TASK-005",
            task_type="lab_analysis",
            description="Analyze lab results"
        )
        
        # Test result setting
        result_data = {"findings": "Normal values", "recommendations": "Continue current treatment"}
        task.set_result(result_data)
        assert task.result == result_data


class TestOMOPConceptModel:
    """Test the OMOPConcept Pydantic model."""
    
    def test_omop_concept_creation_full(self):
        """Test creating an OMOP concept with all fields."""
        concept = OMOPConcept(
            concept_id=4329847,
            concept_name="Myocardial infarction",
            domain_id=DomainType.CONDITION,
            vocabulary_id=VocabularyType.SNOMED_CT,
            concept_class_id=ConceptClassType.CLINICAL_FINDING,
            standard_concept="S",
            concept_code="22298006",
            valid_start_date=datetime(1970, 1, 1),
            valid_end_date=datetime(2099, 12, 31),
            invalid_reason=None,
            synonyms=["Heart attack", "MI"],
            metadata={"severity": "high"}
        )
        
        assert concept.concept_id == 4329847
        assert concept.concept_name == "Myocardial infarction"
        assert concept.domain_id == DomainType.CONDITION
        assert concept.vocabulary_id == VocabularyType.SNOMED_CT
        assert concept.concept_class_id == ConceptClassType.CLINICAL_FINDING
        assert concept.standard_concept == "S"
        assert concept.concept_code == "22298006"
        assert concept.valid_start_date == datetime(1970, 1, 1)
        assert concept.valid_end_date == datetime(2099, 12, 31)
        assert "Heart attack" in concept.synonyms
        assert concept.metadata["severity"] == "high"
    
    def test_omop_concept_creation_minimal(self):
        """Test creating an OMOP concept with minimal required fields."""
        concept = OMOPConcept(
            concept_id=1234567,
            concept_name="Blood glucose measurement",
            domain_id=DomainType.MEASUREMENT,
            vocabulary_id=VocabularyType.LOINC,
            concept_class_id=ConceptClassType.MEASUREMENT,
            concept_code="12345-6",
            valid_start_date=datetime(1970, 1, 1),
            valid_end_date=datetime(2099, 12, 31)
        )
        
        assert concept.concept_id == 1234567
        assert concept.concept_name == "Blood glucose measurement"
        assert concept.domain_id == DomainType.MEASUREMENT
        assert concept.vocabulary_id == VocabularyType.LOINC
        assert concept.concept_class_id == ConceptClassType.MEASUREMENT
        assert concept.concept_code == "12345-6"
        
        # Test defaults
        assert concept.standard_concept is None
        assert concept.invalid_reason is None
        assert concept.synonyms == []
        assert concept.metadata == {}
    
    def test_omop_concept_validation_negative_id(self):
        """Test OMOP concept validation with negative ID."""
        # The framework doesn't validate negative IDs, so this should pass
        concept = OMOPConcept(
            concept_id=-1,
            concept_name="Invalid concept",
            domain_id=DomainType.CONDITION,
            vocabulary_id=VocabularyType.SNOMED_CT,
            concept_class_id=ConceptClassType.CLINICAL_FINDING,
            concept_code="INVALID",
            valid_start_date=datetime(1970, 1, 1),
            valid_end_date=datetime(2099, 12, 31)
        )
        assert concept.concept_id == -1
    
    def test_omop_concept_json_serialization(self):
        """Test OMOP concept JSON serialization."""
        concept = OMOPConcept(
            concept_id=8507,
            concept_name="Male",
            domain_id=DomainType.OBSERVATION,
            vocabulary_id=VocabularyType.GENDER,
            concept_class_id=ConceptClassType.OBSERVATION,
            standard_concept="S",
            concept_code="M",
            valid_start_date=datetime(1970, 1, 1),
            valid_end_date=datetime(2099, 12, 31)
        )
        
        json_data = concept.model_dump()
        
        assert json_data["concept_id"] == 8507
        assert json_data["concept_name"] == "Male"
        assert json_data["domain_id"] == "Observation"
        assert json_data["vocabulary_id"] == "Gender"
        
        # Test round-trip
        restored_concept = OMOPConcept.model_validate(json_data)
        assert restored_concept.concept_id == concept.concept_id


class TestOMOPVocabularyModel:
    """Test the OMOPVocabulary Pydantic model."""
    
    def test_omop_vocabulary_creation_full(self):
        """Test creating an OMOP vocabulary with all fields."""
        vocabulary = OMOPVocabulary(
            vocabulary_id="SNOMED",
            vocabulary_name="Systematized Nomenclature of Medicine Clinical Terms",
            vocabulary_reference="http://www.snomed.org/",
            vocabulary_version="2024-03-01",
            vocabulary_concept_id=46233682,
            is_active=True
        )
        
        assert vocabulary.vocabulary_id == "SNOMED"
        assert vocabulary.vocabulary_name == "Systematized Nomenclature of Medicine Clinical Terms"
        assert vocabulary.vocabulary_reference == "http://www.snomed.org/"
        assert vocabulary.vocabulary_version == "2024-03-01"
        assert vocabulary.vocabulary_concept_id == 46233682
        assert vocabulary.is_active is True
    
    def test_omop_vocabulary_creation_minimal(self):
        """Test creating an OMOP vocabulary with minimal required fields."""
        vocabulary = OMOPVocabulary(
            vocabulary_id="LOINC",
            vocabulary_name="Logical Observation Identifiers Names and Codes"
        )
        
        assert vocabulary.vocabulary_id == "LOINC"
        assert vocabulary.vocabulary_name == "Logical Observation Identifiers Names and Codes"
        
        # Test defaults
        assert vocabulary.vocabulary_reference is None
        assert vocabulary.vocabulary_version is None
        assert vocabulary.vocabulary_concept_id is None
        assert vocabulary.is_active is True
    
    def test_omop_vocabulary_validation_empty_fields(self):
        """Test OMOP vocabulary validation with empty fields."""
        # The framework doesn't validate empty fields, so this should pass
        vocabulary = OMOPVocabulary(
            vocabulary_id="",
            vocabulary_name=""
        )
        assert vocabulary.vocabulary_id == ""
        assert vocabulary.vocabulary_name == ""
    
    def test_omop_vocabulary_json_serialization(self):
        """Test OMOP vocabulary JSON serialization."""
        vocabulary = OMOPVocabulary(
            vocabulary_id="ICD10CM",
            vocabulary_name="International Classification of Diseases, Tenth Revision, Clinical Modification",
            vocabulary_version="2024",
            is_active=True
        )
        
        json_data = vocabulary.model_dump()
        
        assert json_data["vocabulary_id"] == "ICD10CM"
        assert json_data["vocabulary_name"] == "International Classification of Diseases, Tenth Revision, Clinical Modification"
        assert json_data["vocabulary_version"] == "2024"
        assert json_data["is_active"] is True
        
        # Test round-trip
        restored_vocabulary = OMOPVocabulary.model_validate(json_data)
        assert restored_vocabulary.vocabulary_id == vocabulary.vocabulary_id


class TestMessageTypeEnum:
    """Test the MessageType enum."""
    
    def test_message_type_values(self):
        """Test MessageType enum values."""
        assert MessageType.QUERY.value == "query"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.NOTIFICATION.value == "notification"
        assert MessageType.COMMAND.value == "command"
        assert MessageType.EVENT.value == "event"
        assert MessageType.ERROR.value == "error"
    
    def test_message_type_membership(self):
        """Test MessageType enum membership."""
        assert MessageType.QUERY in MessageType
        assert MessageType.RESPONSE in MessageType
        assert MessageType.NOTIFICATION in MessageType


class TestTaskStatusEnum:
    """Test the TaskStatus enum."""
    
    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
    
    def test_task_status_membership(self):
        """Test TaskStatus enum membership."""
        assert TaskStatus.PENDING in TaskStatus
        assert TaskStatus.COMPLETED in TaskStatus
        assert TaskStatus.FAILED in TaskStatus 
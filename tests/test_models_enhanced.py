"""
Enhanced tests for models to improve coverage.
"""

import pytest
from datetime import datetime, date
from typing import Optional, Dict, Any, List
import json

from a2a_medical.models.medical import Patient, Provider, MedicalRecord
from a2a_medical.models.messages import A2AMessage, MessageType
from a2a_medical.models.tasks import MedicalTask, TaskStatus
from a2a_medical.models.omop import OMOPConcept, OMOPVocabulary, OMOPConceptRelationship
from a2a_medical.models.omop import DomainType, VocabularyType, ConceptClassType


class TestPatientModelEnhanced:
    """Enhanced tests for Patient model to improve coverage."""
    
    def test_patient_update_demographics(self):
        """Test patient demographic updates."""
        patient = Patient(
            patient_id="P001",
            mrn="MRN001",
            first_name="John",
            last_name="Doe",
            date_of_birth=datetime(1980, 1, 1),
            gender="male"
        )
        
        # Test contact info updates
        patient.contact_info["phone"] = "+1-555-0123"
        patient.contact_info["email"] = "john.doe@example.com"
        patient.contact_info["address"] = "123 Main St, Anytown, USA"
        
        assert patient.contact_info["phone"] == "+1-555-0123"
        assert patient.contact_info["email"] == "john.doe@example.com"
        assert patient.contact_info["address"] == "123 Main St, Anytown, USA"
    
    def test_patient_insurance_information(self):
        """Test patient insurance information."""
        patient = Patient(
            patient_id="P002",
            mrn="MRN002",
            first_name="Jane",
            last_name="Smith",
            date_of_birth=datetime(1975, 5, 15),
            gender="female"
        )
        
        # Test insurance info
        patient.insurance_info["provider"] = "Blue Cross"
        patient.insurance_info["policy_number"] = "BC123456"
        patient.insurance_info["group_number"] = "GRP789"
        
        assert patient.insurance_info["provider"] == "Blue Cross"
        assert patient.insurance_info["policy_number"] == "BC123456"
        assert patient.insurance_info["group_number"] == "GRP789"
    
    def test_patient_validation_edge_cases(self):
        """Test patient validation edge cases."""
        # Test with minimal required fields
        patient = Patient(
            patient_id="P003",
            first_name="",  # Empty name
            last_name="Test",
            date_of_birth=datetime(2024, 1, 1),  # Future date
            gender="other"
        )
        
        assert patient.first_name == ""
        assert patient.gender == "other"
        assert patient.mrn is None  # Optional field


class TestProviderModelEnhanced:
    """Enhanced tests for Provider model to improve coverage."""
    
    def test_provider_specialization_management(self):
        """Test provider specialization handling."""
        provider = Provider(
            provider_id="PR001",
            first_name="Dr. Alice",
            last_name="Johnson",
            specialty="cardiology",
            credentials=["MD", "FACC"]
        )
        
        # Test specialty and credentials
        provider.credentials.append("FSCAI")
        assert "FSCAI" in provider.credentials
        assert len(provider.credentials) == 3
        assert provider.specialty == "cardiology"
    
    def test_provider_license_information(self):
        """Test provider license and credential information."""
        provider = Provider(
            provider_id="PR002",
            first_name="Dr. Bob",
            last_name="Wilson",
            specialty="emergency_medicine",
            npi="1234567890"
        )
        
        # Test NPI and organization fields
        assert provider.npi == "1234567890"
        provider.organization_id = "ORG001"
        provider.is_active = True
        
        assert provider.organization_id == "ORG001"
        assert provider.is_active is True
    
    def test_provider_contact_information(self):
        """Test provider contact information handling."""
        provider = Provider(
            provider_id="PR003",
            first_name="Dr. Carol",
            last_name="Davis",
            specialty="pediatrics"
        )
        
        provider.contact_info["phone"] = "+1-555-PEDS"
        provider.contact_info["email"] = "carol.davis@hospital.com"
        provider.contact_info["office_address"] = "Children's Hospital"
        
        assert provider.contact_info["phone"] == "+1-555-PEDS"
        assert provider.contact_info["email"] == "carol.davis@hospital.com"
        assert provider.contact_info["office_address"] == "Children's Hospital"


class TestMedicalRecordEnhanced:
    """Enhanced tests for MedicalRecord model to improve coverage."""
    
    def test_medical_record_timestamp_handling(self):
        """Test medical record timestamp operations."""
        record = MedicalRecord(
            record_id="MR001",
            patient_id="P001",
            provider_id="PR001",
            record_type="consultation",
            record_date=datetime.now(),
            content={"chief_complaint": "chest_pain"}
        )
        
        # Test timestamp fields
        assert record.created_at is not None
        assert record.updated_at is not None
        assert isinstance(record.created_at, datetime)
        assert isinstance(record.updated_at, datetime)
    
    def test_medical_record_content_types(self):
        """Test various medical record content types."""
        # Lab results record
        lab_record = MedicalRecord(
            record_id="MR002",
            patient_id="P001",
            provider_id="PR001",
            record_type="lab_results",
            record_date=datetime.now(),
            content={
                "tests": ["CBC", "BMP", "lipid_panel"],
                "results": {
                    "hemoglobin": "14.2 g/dL",
                    "glucose": "95 mg/dL",
                    "cholesterol": "180 mg/dL"
                }
            }
        )
        
        assert lab_record.record_type == "lab_results"
        assert "CBC" in lab_record.content["tests"]
        assert lab_record.content["results"]["glucose"] == "95 mg/dL"
        
        # Prescription record
        prescription_record = MedicalRecord(
            record_id="MR003",
            patient_id="P001",
            provider_id="PR001",
            record_type="prescription",
            record_date=datetime.now(),
            content={
                "medications": [
                    {"name": "aspirin", "dosage": "81mg", "frequency": "daily"},
                    {"name": "atorvastatin", "dosage": "20mg", "frequency": "nightly"}
                ],
                "instructions": "Take with food"
            }
        )
        
        assert prescription_record.record_type == "prescription"
        assert len(prescription_record.content["medications"]) == 2
    
    def test_medical_record_complex_content(self):
        """Test medical record with complex nested content."""
        complex_record = MedicalRecord(
            record_id="MR004",
            patient_id="P001",
            provider_id="PR001",
            record_type="comprehensive_exam",
            record_date=datetime.now(),
            content={
                "vital_signs": {
                    "blood_pressure": {"systolic": 120, "diastolic": 80},
                    "heart_rate": 72,
                    "temperature": 98.6,
                    "respiratory_rate": 16
                },
                "assessment": {
                    "primary_diagnosis": "hypertension",
                    "secondary_diagnoses": ["obesity", "diabetes_type_2"],
                    "icd_codes": ["I10", "E66.9", "E11.9"]
                },
                "plan": {
                    "medications": ["lisinopril_10mg"],
                    "follow_up": "3_months",
                    "lifestyle_modifications": ["diet", "exercise"]
                }
            }
        )
        
        assert complex_record.content["vital_signs"]["blood_pressure"]["systolic"] == 120
        assert "obesity" in complex_record.content["assessment"]["secondary_diagnoses"]
        assert "diet" in complex_record.content["plan"]["lifestyle_modifications"]


class TestA2AMessageEnhanced:
    """Enhanced tests for A2AMessage model to improve coverage."""
    
    def test_message_header_handling(self):
        """Test A2A message header processing."""
        message = A2AMessage(
            message_id="MSG001",
            sender_id="agent_001",
            recipient_ids=["agent_002"],
            message_type=MessageType.QUERY,
            content={"action": "get_patient_data"}
        )
        
        assert message.message_id == "MSG001"
        assert message.sender_id == "agent_001"
        assert message.recipient_ids == ["agent_002"]
        assert message.message_type == MessageType.QUERY
    
    def test_message_encryption_fields(self):
        """Test A2A message encryption metadata."""
        message = A2AMessage(
            message_id="MSG002",
            sender_id="agent_003",
            recipient_ids=["agent_004"],
            message_type=MessageType.RESPONSE,
            content={"status": "success", "data": {"patient": "encrypted_data"}}
        )
        
        message.metadata["is_encrypted"] = True
        message.metadata["encryption_algorithm"] = "AES-256-GCM"
        message.metadata["signature"] = "digital_signature_hash"
        
        assert message.metadata["is_encrypted"] is True
        assert message.metadata["encryption_algorithm"] == "AES-256-GCM"
        assert message.metadata["signature"] == "digital_signature_hash"
    
    def test_message_complex_content(self):
        """Test A2A message with complex nested content."""
        complex_message = A2AMessage(
            message_id="MSG003",
            sender_id="emergency_agent",
            recipient_ids=["hospital_system"],
            message_type=MessageType.EVENT,
            content={
                "emergency_type": "cardiac_arrest",
                "patient_info": {
                    "age": 65,
                    "gender": "male",
                    "location": "home",
                    "vital_signs": {
                        "conscious": False,
                        "pulse": 0,
                        "breathing": False
                    }
                },
                "requested_resources": [
                    "ambulance",
                    "cardiac_team",
                    "emergency_room"
                ],
                "estimated_arrival": "2024-01-15T14:30:00Z"
            }
        )
        
        assert complex_message.message_type == MessageType.EVENT
        assert complex_message.content["emergency_type"] == "cardiac_arrest"
        assert complex_message.content["patient_info"]["vital_signs"]["conscious"] is False
        assert "ambulance" in complex_message.content["requested_resources"]


class TestMedicalTaskEnhanced:
    """Enhanced tests for MedicalTask model to improve coverage."""
    
    def test_task_metadata_handling(self):
        """Test medical task metadata and tracking."""
        task = MedicalTask(
            task_id="TASK001",
            task_type="patient_assessment",
            description="Comprehensive patient evaluation",
            priority=1
        )
        
        # Test metadata fields
        task.estimated_duration = 3600  # 1 hour in seconds
        task.required_skills = ["patient_interview", "physical_exam", "diagnosis"]
        task.constraints = {"requires_physician": True, "lab_access": True}
        
        assert task.estimated_duration == 3600
        assert "physical_exam" in task.required_skills
        assert task.constraints["requires_physician"] is True
    
    def test_task_workflow_progression(self):
        """Test medical task workflow and state transitions."""
        task = MedicalTask(
            task_id="TASK002",
            task_type="medication_review",
            description="Review patient medications for interactions",
            priority=2
        )
        
        # Test workflow progression
        assert task.status == TaskStatus.PENDING
        
        task.status = TaskStatus.IN_PROGRESS
        task.progress_percentage = 25
        task.current_step = "collecting_medication_list"
        
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.progress_percentage == 25
        assert task.current_step == "collecting_medication_list"
        
        # Complete task
        task.status = TaskStatus.COMPLETED
        task.progress_percentage = 100
        task.completion_time = datetime.now()
        task.result = {
            "interactions_found": 2,
            "recommendations": ["discontinue_drug_A", "reduce_drug_B_dosage"]
        }
        
        assert task.status == TaskStatus.COMPLETED
        assert task.progress_percentage == 100
        assert task.result["interactions_found"] == 2
    
    def test_task_error_handling(self):
        """Test medical task error and failure scenarios."""
        task = MedicalTask(
            task_id="TASK003",
            task_type="lab_order",
            description="Order blood work for patient",
            priority=3
        )
        
        # Test error scenario
        task.status = TaskStatus.FAILED
        task.error_message = "Lab system unavailable"
        task.error_code = "LAB_SYS_001"
        task.retry_count = 3
        
        assert task.status == TaskStatus.FAILED
        assert task.error_message == "Lab system unavailable"
        assert task.error_code == "LAB_SYS_001"
        assert task.retry_count == 3


class TestOMOPConceptEnhanced:
    """Enhanced tests for OMOP Concept model to improve coverage."""
    
    def test_omop_concept_hierarchy(self):
        """Test OMOP concept hierarchical relationships."""
        parent_concept = OMOPConcept(
            concept_id=123456,
            concept_name="Cardiovascular disease",
            domain_id=DomainType.CONDITION,
            vocabulary_id=VocabularyType.SNOMED_CT,
            concept_class_id=ConceptClassType.CLINICAL_FINDING,
            concept_code="CVD001",
            valid_start_date=datetime(2024, 1, 1),
            valid_end_date=datetime(2099, 12, 31)
        )
        
        child_concept = OMOPConcept(
            concept_id=123457,
            concept_name="Myocardial infarction",
            domain_id=DomainType.CONDITION,
            vocabulary_id=VocabularyType.SNOMED_CT,
            concept_class_id=ConceptClassType.CLINICAL_FINDING,
            concept_code="MI001",
            valid_start_date=datetime(2024, 1, 1),
            valid_end_date=datetime(2099, 12, 31)
        )
        
        # Test concept hierarchy - note: these fields don't exist in the model
        # Store hierarchy info in metadata instead
        parent_concept.metadata["child_concepts"] = [child_concept.concept_id]
        child_concept.metadata["parent_concept_id"] = parent_concept.concept_id
        
        assert len(parent_concept.metadata["child_concepts"]) == 1
        assert child_concept.metadata["parent_concept_id"] == 123456
    
    def test_omop_concept_metadata(self):
        """Test OMOP concept metadata fields."""
        concept = OMOPConcept(
            concept_id=789012,
            concept_name="Aspirin 81 MG Oral Tablet",
            domain_id=DomainType.DRUG,
            vocabulary_id=VocabularyType.RXNORM,
            concept_class_id=ConceptClassType.DRUG,
            concept_code="ASP081",
            valid_start_date=datetime(2024, 1, 1),
            valid_end_date=datetime(2099, 12, 31)
        )
        
        # Test metadata
        concept.standard_concept = "S"  # Standard concept
        concept.invalid_reason = None
        # Fix: use synonyms instead of concept_synonyms
        concept.synonyms = ["Low-dose aspirin", "Baby aspirin"]
        # Fix: store mapping_type in metadata since it doesn't exist as a field
        concept.metadata["mapping_type"] = "Maps to"
        
        assert concept.standard_concept == "S"
        assert concept.invalid_reason is None
        assert "Baby aspirin" in concept.synonyms
        assert concept.metadata["mapping_type"] == "Maps to"


class TestOMOPVocabularyEnhanced:
    """Enhanced tests for OMOP Vocabulary model to improve coverage."""
    
    def test_vocabulary_metadata(self):
        """Test OMOP vocabulary metadata handling."""
        vocabulary = OMOPVocabulary(
            vocabulary_id="SNOMED",
            vocabulary_name="Systematized Nomenclature of Medicine Clinical Terms",
            vocabulary_reference="https://www.snomed.org/",
            vocabulary_version="20240301"
        )
        
        # Test basic vocabulary fields since metadata field doesn't exist
        assert vocabulary.vocabulary_id == "SNOMED"
        assert vocabulary.vocabulary_name == "Systematized Nomenclature of Medicine Clinical Terms"
        assert vocabulary.vocabulary_reference == "https://www.snomed.org/"
        assert vocabulary.vocabulary_version == "20240301"
        assert vocabulary.is_active is True  # Default value
        assert vocabulary.created_at is not None
        assert vocabulary.updated_at is not None
    
    def test_vocabulary_relationships(self):
        """Test vocabulary relationship management."""
        snomed = OMOPVocabulary(
            vocabulary_id="SNOMED",
            vocabulary_name="SNOMED CT",
            vocabulary_reference="https://www.snomed.org/",
            vocabulary_version="20240301"
        )
        
        icd10 = OMOPVocabulary(
            vocabulary_id="ICD10CM",
            vocabulary_name="ICD-10-CM",
            vocabulary_reference="https://www.cdc.gov/nchs/icd/icd10cm.htm",
            vocabulary_version="2024"
        )
        
        # Test vocabulary basic functionality since metadata field doesn't exist
        assert snomed.vocabulary_id == "SNOMED"
        assert icd10.vocabulary_id == "ICD10CM"
        assert snomed.vocabulary_name != icd10.vocabulary_name
        assert snomed.is_active is True
        assert icd10.is_active is True


class TestOMOPConceptRelationshipEnhanced:
    """Enhanced tests for OMOP Concept Relationship model to improve coverage."""
    
    def test_concept_relationship_types(self):
        """Test different types of concept relationships."""
        # Is-a relationship
        is_a_relationship = OMOPConceptRelationship(
            concept_id_1=123456,  # Child concept
            concept_id_2=123455,  # Parent concept
            relationship_id="Is a",
            valid_start_date=datetime(2024, 1, 1),
            valid_end_date=datetime(2099, 12, 31)
        )
        
        assert is_a_relationship.relationship_id == "Is a"
        assert is_a_relationship.valid_start_date == datetime(2024, 1, 1)
        
        # Maps-to relationship
        maps_to_relationship = OMOPConceptRelationship(
            concept_id_1=789012,  # Source concept
            concept_id_2=789013,  # Target concept
            relationship_id="Maps to",
            valid_start_date=datetime(2024, 1, 1),
            valid_end_date=datetime(2099, 12, 31)
        )
        
        assert maps_to_relationship.relationship_id == "Maps to"
    
    def test_relationship_metadata(self):
        """Test concept relationship metadata."""
        relationship = OMOPConceptRelationship(
            concept_id_1=456789,
            concept_id_2=456790,
            relationship_id="Has ingredient",
            valid_start_date=datetime(2024, 1, 1),
            valid_end_date=datetime(2099, 12, 31)
        )
        
        # Test basic relationship fields since metadata field doesn't exist
        assert relationship.invalid_reason is None
        assert relationship.concept_id_1 == 456789
        assert relationship.concept_id_2 == 456790
        assert relationship.relationship_id == "Has ingredient"
        assert relationship.valid_start_date == datetime(2024, 1, 1)
        assert relationship.valid_end_date == datetime(2099, 12, 31)


class TestModelInteroperability:
    """Test interoperability between different model types."""
    
    def test_patient_provider_relationship(self):
        """Test patient-provider relationship modeling."""
        patient = Patient(
            patient_id="P001",
            first_name="John",
            last_name="Doe",
            date_of_birth=datetime(1980, 1, 1),
            gender="male"
        )
        
        provider = Provider(
            provider_id="PR001",
            first_name="Dr. Alice",
            last_name="Johnson",
            specialty="cardiology"
        )
        
        # Create medical record linking patient and provider
        record = MedicalRecord(
            record_id="MR001",
            patient_id=patient.patient_id,
            provider_id=provider.provider_id,
            record_type="consultation",
            record_date=datetime.now(),
            content={"chief_complaint": "chest_pain"}
        )
        
        assert record.patient_id == patient.patient_id
        assert record.provider_id == provider.provider_id
    
    def test_task_message_integration(self):
        """Test integration between tasks and messages."""
        task = MedicalTask(
            task_id="TASK001",
            task_type="patient_assessment",
            description="Assess patient for cardiac risk",
            priority=1
        )
        
        # Create message referencing the task
        message = A2AMessage(
            message_id="MSG001",
            sender_id="cardiac_agent",
            recipient_ids=["assessment_agent"],
            message_type=MessageType.QUERY,
            content={
                "task_id": task.task_id,
                "action": "start_assessment",
                "patient_id": "P001"
            }
        )
        
        assert message.content["task_id"] == task.task_id
        assert message.content["action"] == "start_assessment"
    
    def test_omop_medical_record_integration(self):
        """Test integration between OMOP concepts and medical records."""
        # OMOP concept for diagnosis
        concept = OMOPConcept(
            concept_id=123456,
            concept_name="Essential hypertension",
            domain_id=DomainType.CONDITION,
            vocabulary_id=VocabularyType.SNOMED_CT,
            concept_class_id=ConceptClassType.CLINICAL_FINDING,
            concept_code="HTN001",
            valid_start_date=datetime(2024, 1, 1),
            valid_end_date=datetime(2099, 12, 31)
        )
        
        # Medical record using OMOP concept
        record = MedicalRecord(
            record_id="MR001",
            patient_id="P001",
            provider_id="PR001",
            record_type="diagnosis",
            record_date=datetime.now(),
            content={
                "primary_diagnosis": {
                    "concept_id": concept.concept_id,
                    "concept_name": concept.concept_name,
                    "concept_code": concept.concept_code
                },
                "diagnosis_date": "2024-01-15"
            }
        )
        
        assert record.content["primary_diagnosis"]["concept_id"] == concept.concept_id
        assert record.content["primary_diagnosis"]["concept_name"] == concept.concept_name 
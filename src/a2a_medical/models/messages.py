from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from .medical import MedicalQuery


class MessageType(str, Enum):
    """Types of A2A messages."""
    QUERY = "query"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    COMMAND = "command"
    EVENT = "event"
    ERROR = "error"


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class MessageStatus(str, Enum):
    """Message status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    EXPIRED = "expired"


class MessagePart(BaseModel):
    """A part of an A2A message."""
    part_id: str = Field(..., description="Unique part identifier")
    content_type: str = Field(..., description="MIME type of the content")
    content: Any = Field(..., description="Message content")
    encoding: Optional[str] = Field(None, description="Content encoding")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MedicalMessageMetadata(BaseModel):
    """Medical-specific metadata for A2A messages."""
    medical_query: Optional[MedicalQuery] = None
    compliance_flags: List[str] = Field(default_factory=list)
    phi_present: bool = False
    requires_audit: bool = True
    clinical_priority: int = Field(1, ge=1, le=5)
    patient_context: Optional[Dict[str, Any]] = None
    provider_context: Optional[Dict[str, Any]] = None


class A2AMessage(BaseModel):
    """Base A2A message model."""
    message_id: str = Field(..., description="Unique message identifier")
    message_type: MessageType = Field(..., description="Type of message")
    sender_id: str = Field(..., description="Sender agent ID")
    recipient_ids: List[str] = Field(..., description="Recipient agent IDs")
    content: Dict[str, Any] = Field(..., description="Message content")
    parts: List[MessagePart] = Field(default_factory=list)
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_part(self, part: MessagePart) -> None:
        """Add a part to the message."""
        self.parts.append(part)
    
    def get_part(self, part_id: str) -> Optional[MessagePart]:
        """Get a specific part by ID."""
        for part in self.parts:
            if part.part_id == part_id:
                return part
        return None
    
    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def mark_sent(self) -> None:
        """Mark the message as sent."""
        self.status = MessageStatus.SENT
        self.sent_at = datetime.utcnow()
    
    def mark_delivered(self) -> None:
        """Mark the message as delivered."""
        self.status = MessageStatus.DELIVERED
        self.delivered_at = datetime.utcnow()
    
    def mark_failed(self) -> None:
        """Mark the message as failed."""
        self.status = MessageStatus.FAILED


class MedicalMessage(A2AMessage):
    """Extended A2A message for medical domain."""
    medical_metadata: Optional[MedicalMessageMetadata] = None
    
    def contains_phi(self) -> bool:
        """Check if message contains Protected Health Information."""
        return bool(self.medical_metadata and self.medical_metadata.phi_present)
    
    def requires_audit_trail(self) -> bool:
        """Check if message requires audit trail."""
        return bool(self.medical_metadata and self.medical_metadata.requires_audit)
    
    def get_clinical_priority(self) -> int:
        """Get the clinical priority level."""
        if self.medical_metadata:
            return self.medical_metadata.clinical_priority
        return 1
    
    def add_compliance_flag(self, flag: str) -> None:
        """Add a compliance flag to the message."""
        if self.medical_metadata:
            self.medical_metadata.compliance_flags.append(flag)
        else:
            self.medical_metadata = MedicalMessageMetadata(
                compliance_flags=[flag],
                clinical_priority=1
            )


class MessageFactory:
    """Factory for creating A2A messages."""
    
    @staticmethod
    def create_query_message(sender_id: str, recipient_ids: List[str], 
                           query_content: Dict[str, Any], 
                           medical_query: Optional[MedicalQuery] = None) -> MedicalMessage:
        """Create a medical query message."""
        message = MedicalMessage(
            message_id=f"query_{datetime.utcnow().timestamp()}",
            message_type=MessageType.QUERY,
            sender_id=sender_id,
            recipient_ids=recipient_ids,
            content=query_content
        )
        
        if medical_query:
            message.medical_metadata = MedicalMessageMetadata(
                medical_query=medical_query,
                clinical_priority=medical_query.urgency_level
            )
        
        return message
    
    @staticmethod
    def create_response_message(sender_id: str, recipient_ids: List[str],
                              response_content: Dict[str, Any],
                              original_message_id: str) -> MedicalMessage:
        """Create a medical response message."""
        message = MedicalMessage(
            message_id=f"response_{datetime.utcnow().timestamp()}",
            message_type=MessageType.RESPONSE,
            sender_id=sender_id,
            recipient_ids=recipient_ids,
            content=response_content,
            metadata={"original_message_id": original_message_id}
        )
        
        return message
    
    @staticmethod
    def create_notification_message(sender_id: str, recipient_ids: List[str],
                                  notification_content: Dict[str, Any],
                                  priority: MessagePriority = MessagePriority.NORMAL) -> MedicalMessage:
        """Create a medical notification message."""
        message = MedicalMessage(
            message_id=f"notification_{datetime.utcnow().timestamp()}",
            message_type=MessageType.NOTIFICATION,
            sender_id=sender_id,
            recipient_ids=recipient_ids,
            content=notification_content,
            priority=priority
        )
        
        return message
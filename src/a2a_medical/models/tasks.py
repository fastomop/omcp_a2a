"""
Task management models for medical A2A operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """Status enumeration for medical tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Priority levels for medical tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class MedicalTask:
    """Represents a medical task to be executed by an agent."""
    
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    result: Optional[Any] = None
    
    def __post_init__(self):
        """Update the updated_at timestamp after initialization."""
        self.updated_at = datetime.utcnow()
    
    def update_status(self, status: TaskStatus, error_message: Optional[str] = None) -> None:
        """Update the task status."""
        self.status = status
        self.updated_at = datetime.utcnow()
        
        if status == TaskStatus.COMPLETED:
            self.completed_at = datetime.utcnow()
        
        if error_message:
            self.error_message = error_message
    
    def assign_to_agent(self, agent_id: str) -> None:
        """Assign the task to a specific agent."""
        self.assigned_agent = agent_id
        self.updated_at = datetime.utcnow()
    
    def set_result(self, result: Any) -> None:
        """Set the task result."""
        self.result = result
        self.updated_at = datetime.utcnow()
    
    def is_completed(self) -> bool:
        """Check if the task is completed."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
    
    def is_active(self) -> bool:
        """Check if the task is active (not completed)."""
        return not self.is_completed()
    
    def get_duration(self) -> Optional[float]:
        """Get the task duration in seconds."""
        if self.completed_at and self.created_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None


@dataclass
class TaskBatch:
    """Represents a batch of related medical tasks."""
    
    batch_id: str
    tasks: List[MedicalTask] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_task(self, task: MedicalTask) -> None:
        """Add a task to the batch."""
        self.tasks.append(task)
    
    def get_completed_tasks(self) -> List[MedicalTask]:
        """Get all completed tasks in the batch."""
        return [task for task in self.tasks if task.is_completed()]
    
    def get_pending_tasks(self) -> List[MedicalTask]:
        """Get all pending tasks in the batch."""
        return [task for task in self.tasks if not task.is_completed()]
    
    def get_batch_status(self) -> TaskStatus:
        """Get the overall status of the batch."""
        if not self.tasks:
            return TaskStatus.PENDING
        
        completed_count = len(self.get_completed_tasks())
        total_count = len(self.tasks)
        
        if completed_count == total_count:
            return TaskStatus.COMPLETED
        elif completed_count > 0:
            return TaskStatus.IN_PROGRESS
        else:
            return TaskStatus.PENDING

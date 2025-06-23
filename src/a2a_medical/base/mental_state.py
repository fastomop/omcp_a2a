"""
Mental state management for medical agents.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


class MentalState:
    """Represents the mental state of a medical agent."""
    
    def __init__(self):
        self.beliefs: Dict[str, Any] = {}
        self.goals: List[str] = []
        self.emotions: Dict[str, float] = {}
        self.memory: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        self.last_updated: datetime = datetime.utcnow()
    
    def update_belief(self, key: str, value: Any) -> None:
        """Update a belief in the mental state."""
        self.beliefs[key] = value
        self.last_updated = datetime.utcnow()
    
    def add_goal(self, goal: str) -> None:
        """Add a goal to the mental state."""
        if goal not in self.goals:
            self.goals.append(goal)
        self.last_updated = datetime.utcnow()
    
    def set_emotion(self, emotion: str, intensity: float) -> None:
        """Set an emotion intensity in the mental state."""
        self.emotions[emotion] = max(0.0, min(1.0, intensity))
        self.last_updated = datetime.utcnow()
    
    def add_to_memory(self, key: str, value: Any) -> None:
        """Add information to memory."""
        self.memory[key] = value
        self.last_updated = datetime.utcnow()
    
    def get_belief(self, key: str) -> Optional[Any]:
        """Get a belief from the mental state."""
        return self.beliefs.get(key)
    
    def has_goal(self, goal: str) -> bool:
        """Check if a goal exists in the mental state."""
        return goal in self.goals
    
    def get_emotion(self, emotion: str) -> float:
        """Get the intensity of an emotion."""
        return self.emotions.get(emotion, 0.0)
    
    def get_memory(self, key: str) -> Optional[Any]:
        """Get information from memory."""
        return self.memory.get(key)
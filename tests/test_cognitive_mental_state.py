"""
Tests for cognitive modules and mental state components.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from abc import ABC

from a2a_medical.base.cognitive import CognitiveModule, ReasoningModule, LearningModule
from a2a_medical.base.mental_state import MentalState


class TestCognitiveModuleAbstract:
    """Test the abstract CognitiveModule class."""
    
    def test_cannot_instantiate_abstract_cognitive_module(self):
        """Test that CognitiveModule cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CognitiveModule("test-module")
    
    def test_cognitive_module_has_required_abstract_methods(self):
        """Test that CognitiveModule has required abstract methods."""
        abstract_methods = CognitiveModule.__abstractmethods__
        expected_methods = {"process", "learn"}
        assert abstract_methods == expected_methods


class TestCognitiveModuleConcrete:
    """Test concrete CognitiveModule implementations."""
    
    def test_cognitive_module_initialization(self):
        """Test CognitiveModule initialization."""
        
        class TestCognitiveModule(CognitiveModule):
            async def process(self, mental_state, input_data):
                return {"result": "processed"}
            
            async def learn(self, mental_state, feedback):
                pass
        
        module = TestCognitiveModule("test-module")
        
        assert module.module_name == "test-module"
        assert module.is_active is True
    
    def test_cognitive_module_activation_deactivation(self):
        """Test CognitiveModule activation and deactivation."""
        
        class TestCognitiveModule(CognitiveModule):
            async def process(self, mental_state, input_data):
                return {"result": "processed"}
            
            async def learn(self, mental_state, feedback):
                pass
        
        module = TestCognitiveModule("test-module")
        
        # Test initial state
        assert module.is_active is True
        
        # Test deactivation
        module.deactivate()
        assert module.is_active is False
        
        # Test activation
        module.activate()
        assert module.is_active is True
    
    def test_cognitive_module_status(self):
        """Test CognitiveModule status retrieval."""
        
        class TestCognitiveModule(CognitiveModule):
            async def process(self, mental_state, input_data):
                return {"result": "processed"}
            
            async def learn(self, mental_state, feedback):
                pass
        
        module = TestCognitiveModule("test-module")
        status = module.get_status()
        
        assert status["module_name"] == "test-module"
        assert status["is_active"] is True
        
        module.deactivate()
        status = module.get_status()
        assert status["is_active"] is False


class TestReasoningModule:
    """Test the ReasoningModule concrete implementation."""
    
    def test_reasoning_module_initialization(self):
        """Test ReasoningModule initialization."""
        module = ReasoningModule()
        
        assert module.module_name == "reasoning"
        assert module.is_active is True
        assert module.reasoning_rules == []
    
    @pytest.mark.asyncio
    async def test_reasoning_module_process(self):
        """Test ReasoningModule process method."""
        module = ReasoningModule()
        mental_state = MentalState()
        input_data = {"symptom": "fever", "severity": "high"}
        
        result = await module.process(mental_state, input_data)
        
        assert result["reasoning_result"] == "processed"
        assert result["confidence"] == 0.8
        assert result["rules_applied"] == 0  # No rules added yet
    
    @pytest.mark.asyncio
    async def test_reasoning_module_learn(self):
        """Test ReasoningModule learn method."""
        module = ReasoningModule()
        mental_state = MentalState()
        feedback = {"accuracy": 0.9, "improvement": "add_rule"}
        
        # Should not raise any exceptions
        await module.learn(mental_state, feedback)
        assert True  # Placeholder implementation doesn't modify state
    
    def test_reasoning_module_with_rules(self):
        """Test ReasoningModule with rules added."""
        module = ReasoningModule()
        
        # Add some reasoning rules
        module.reasoning_rules.append({"condition": "fever", "action": "recommend_fluids"})
        module.reasoning_rules.append({"condition": "high_temp", "action": "urgent_care"})
        
        assert len(module.reasoning_rules) == 2


class TestLearningModule:
    """Test the LearningModule concrete implementation."""
    
    def test_learning_module_initialization(self):
        """Test LearningModule initialization."""
        module = LearningModule()
        
        assert module.module_name == "learning"
        assert module.is_active is True
        assert module.learning_patterns == {}
    
    @pytest.mark.asyncio
    async def test_learning_module_process(self):
        """Test LearningModule process method."""
        module = LearningModule()
        mental_state = MentalState()
        input_data = {"patient_age": 45, "symptoms": ["fatigue", "headache"]}
        
        result = await module.process(mental_state, input_data)
        
        assert result["learning_result"] == "patterns_identified"
        assert result["new_patterns"] == 0
        assert result["confidence"] == 0.7
    
    @pytest.mark.asyncio
    async def test_learning_module_learn(self):
        """Test LearningModule learn method."""
        module = LearningModule()
        mental_state = MentalState()
        feedback = {"pattern_accuracy": 0.85, "new_pattern": True}
        
        # Should not raise any exceptions
        await module.learn(mental_state, feedback)
        assert True  # Placeholder implementation doesn't modify state
    
    def test_learning_module_with_patterns(self):
        """Test LearningModule with patterns added."""
        module = LearningModule()
        
        # Add some learning patterns
        module.learning_patterns["fever_pattern"] = {"trigger": "temp > 38", "response": "fever_detected"}
        module.learning_patterns["pain_pattern"] = {"trigger": "pain_scale > 7", "response": "high_pain"}
        
        assert len(module.learning_patterns) == 2
        assert "fever_pattern" in module.learning_patterns


class TestMentalState:
    """Test the MentalState class."""
    
    def test_mental_state_initialization(self):
        """Test MentalState initialization."""
        mental_state = MentalState()
        
        assert mental_state.beliefs == {}
        assert mental_state.goals == []
        assert mental_state.emotions == {}
        assert mental_state.memory == {}
        assert mental_state.context == {}
        assert isinstance(mental_state.last_updated, datetime)
    
    def test_mental_state_beliefs(self):
        """Test MentalState belief management."""
        mental_state = MentalState()
        
        # Test updating beliefs
        mental_state.update_belief("patient_diagnosis", "hypertension")
        mental_state.update_belief("confidence_level", 0.8)
        
        assert mental_state.get_belief("patient_diagnosis") == "hypertension"
        assert mental_state.get_belief("confidence_level") == 0.8
        assert mental_state.get_belief("nonexistent") is None
        
        # Test belief updates
        mental_state.update_belief("patient_diagnosis", "diabetes")
        assert mental_state.get_belief("patient_diagnosis") == "diabetes"
    
    def test_mental_state_goals(self):
        """Test MentalState goal management."""
        mental_state = MentalState()
        
        # Test adding goals
        mental_state.add_goal("diagnose_condition")
        mental_state.add_goal("provide_treatment_plan")
        
        assert mental_state.has_goal("diagnose_condition")
        assert mental_state.has_goal("provide_treatment_plan")
        assert not mental_state.has_goal("nonexistent_goal")
        assert len(mental_state.goals) == 2
        
        # Test adding duplicate goal
        mental_state.add_goal("diagnose_condition")
        assert len(mental_state.goals) == 2  # Should not add duplicate
    
    def test_mental_state_emotions(self):
        """Test MentalState emotion management."""
        mental_state = MentalState()
        
        # Test setting emotions
        mental_state.set_emotion("confidence", 0.8)
        mental_state.set_emotion("concern", 0.3)
        
        assert mental_state.get_emotion("confidence") == 0.8
        assert mental_state.get_emotion("concern") == 0.3
        assert mental_state.get_emotion("nonexistent") == 0.0
        
        # Test emotion bounds (should clamp to [0.0, 1.0])
        mental_state.set_emotion("excitement", 1.5)  # Should be clamped to 1.0
        mental_state.set_emotion("worry", -0.5)      # Should be clamped to 0.0
        
        assert mental_state.get_emotion("excitement") == 1.0
        assert mental_state.get_emotion("worry") == 0.0
    
    def test_mental_state_memory(self):
        """Test MentalState memory management."""
        mental_state = MentalState()
        
        # Test adding to memory
        mental_state.add_to_memory("patient_history", {"allergies": ["penicillin"], "medications": []})
        mental_state.add_to_memory("last_consultation", "2024-01-15")
        
        assert mental_state.get_memory("patient_history") == {"allergies": ["penicillin"], "medications": []}
        assert mental_state.get_memory("last_consultation") == "2024-01-15"
        assert mental_state.get_memory("nonexistent") is None
        
        # Test memory updates
        mental_state.add_to_memory("patient_history", {"allergies": ["penicillin", "sulfa"], "medications": ["aspirin"]})
        updated_history = mental_state.get_memory("patient_history")
        assert "sulfa" in updated_history["allergies"]
        assert "aspirin" in updated_history["medications"]
    
    def test_mental_state_timestamp_updates(self):
        """Test that MentalState operations update timestamps."""
        mental_state = MentalState()
        initial_timestamp = mental_state.last_updated
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        # Test that operations update timestamps
        mental_state.update_belief("test", "value")
        assert mental_state.last_updated > initial_timestamp
        
        time.sleep(0.001)
        belief_timestamp = mental_state.last_updated
        
        mental_state.add_goal("test_goal")
        assert mental_state.last_updated > belief_timestamp
        
        time.sleep(0.001)
        goal_timestamp = mental_state.last_updated
        
        mental_state.set_emotion("happiness", 0.7)
        assert mental_state.last_updated > goal_timestamp
        
        time.sleep(0.001)
        emotion_timestamp = mental_state.last_updated
        
        mental_state.add_to_memory("test_memory", "test_value")
        assert mental_state.last_updated > emotion_timestamp
    
    def test_mental_state_context(self):
        """Test MentalState context management."""
        mental_state = MentalState()
        
        # Test context manipulation
        mental_state.context["current_session"] = "urgent_care_2024_01_15"
        mental_state.context["agent_mode"] = "emergency"
        mental_state.context["collaboration_agents"] = ["dr_smith_agent", "nurse_jones_agent"]
        
        assert mental_state.context["current_session"] == "urgent_care_2024_01_15"
        assert mental_state.context["agent_mode"] == "emergency"
        assert len(mental_state.context["collaboration_agents"]) == 2


class TestCognitiveModuleIntegration:
    """Test integration between cognitive modules and mental state."""
    
    @pytest.mark.asyncio
    async def test_reasoning_module_with_mental_state(self):
        """Test ReasoningModule integration with MentalState."""
        reasoning_module = ReasoningModule()
        mental_state = MentalState()
        
        # Set up mental state
        mental_state.update_belief("patient_temp", 39.5)
        mental_state.add_goal("assess_fever")
        mental_state.set_emotion("concern", 0.7)
        
        # Process with reasoning module
        input_data = {"new_symptom": "chills", "duration": "2_hours"}
        result = await reasoning_module.process(mental_state, input_data)
        
        assert result is not None
        assert "reasoning_result" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_learning_module_with_mental_state(self):
        """Test LearningModule integration with MentalState."""
        learning_module = LearningModule()
        mental_state = MentalState()
        
        # Set up mental state with learning context
        mental_state.add_to_memory("past_cases", [{"symptoms": ["fever", "cough"], "diagnosis": "flu"}])
        mental_state.context["learning_mode"] = "active"
        
        # Process with learning module
        input_data = {"current_case": {"symptoms": ["fever", "headache"], "age": 30}}
        result = await learning_module.process(mental_state, input_data)
        
        assert result is not None
        assert "learning_result" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_multiple_cognitive_modules(self):
        """Test multiple cognitive modules working with the same mental state."""
        reasoning_module = ReasoningModule()
        learning_module = LearningModule()
        mental_state = MentalState()
        
        # Set up mental state
        mental_state.update_belief("patient_age", 45)
        mental_state.add_goal("comprehensive_assessment")
        
        # Process with both modules
        input_data = {"symptoms": ["fatigue", "joint_pain"], "duration": "3_weeks"}
        
        reasoning_result = await reasoning_module.process(mental_state, input_data)
        learning_result = await learning_module.process(mental_state, input_data)
        
        assert reasoning_result["reasoning_result"] == "processed"
        assert learning_result["learning_result"] == "patterns_identified"
        
        # Both modules should work independently
        assert reasoning_result != learning_result


class TestCognitiveModuleErrorHandling:
    """Test error handling in cognitive modules."""
    
    @pytest.mark.asyncio
    async def test_reasoning_module_with_invalid_input(self):
        """Test ReasoningModule with invalid input data."""
        reasoning_module = ReasoningModule()
        mental_state = MentalState()
        
        # Test with None input
        result = await reasoning_module.process(mental_state, None)
        assert result is not None  # Should handle gracefully
        
        # Test with empty input
        result = await reasoning_module.process(mental_state, {})
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_learning_module_with_invalid_input(self):
        """Test LearningModule with invalid input data."""
        learning_module = LearningModule()
        mental_state = MentalState()
        
        # Test with None input
        result = await learning_module.process(mental_state, None)
        assert result is not None  # Should handle gracefully
        
        # Test with empty input
        result = await learning_module.process(mental_state, {})
        assert result is not None
    
    def test_mental_state_with_invalid_emotions(self):
        """Test MentalState emotion bounds enforcement."""
        mental_state = MentalState()
        
        # Test extreme values
        mental_state.set_emotion("test_high", 999.9)
        mental_state.set_emotion("test_low", -999.9)
        
        assert mental_state.get_emotion("test_high") == 1.0
        assert mental_state.get_emotion("test_low") == 0.0
        
        # Test edge cases
        mental_state.set_emotion("test_edge_high", 1.0)
        mental_state.set_emotion("test_edge_low", 0.0)
        
        assert mental_state.get_emotion("test_edge_high") == 1.0
        assert mental_state.get_emotion("test_edge_low") == 0.0 
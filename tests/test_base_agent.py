"""
Tests for base agent components: WorldModel, MedicalAgent, MentalState, and CognitiveModule.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from abc import ABC

from a2a_medical.base.agent import (
    WorldModel, MedicalAgent, MentalState, CognitiveModule,
    ProcessedObservation, Action, ActionResult
)


class TestWorldModelAbstract:
    """Test the abstract WorldModel class."""
    
    def test_cannot_instantiate_abstract_world_model(self):
        """Test that WorldModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            WorldModel()
    
    def test_world_model_has_required_abstract_methods(self):
        """Test that WorldModel has all required abstract methods."""
        abstract_methods = WorldModel.__abstractmethods__
        expected_methods = {"update", "query", "predict", "get_state_summary", "reset"}
        assert abstract_methods == expected_methods


class TestWorldModelConcrete:
    """Test concrete WorldModel implementations."""
    
    def test_world_model_initialization(self, test_world_model):
        """Test WorldModel initialization."""
        assert test_world_model.model_id is not None
        assert test_world_model.created_at is not None
        assert test_world_model.last_updated is not None
        assert test_world_model.agent_registry == {}
        assert test_world_model.collaboration_history == []
    
    def test_world_model_update(self, test_world_model):
        """Test WorldModel update functionality."""
        observation = ProcessedObservation(
            data={"type": "patient_vitals"},
            timestamp=1234567890.0,
            source="test",
            confidence=0.9
        )
        
        test_world_model.update(observation)
        assert len(test_world_model.test_observations) == 1
        assert test_world_model.test_observations[0] == observation
    
    def test_world_model_query(self, test_world_model):
        """Test WorldModel query functionality."""
        result = test_world_model.query("test_query")
        assert result == "test_result"
    
    def test_world_model_predict(self, test_world_model):
        """Test WorldModel prediction functionality."""
        result = test_world_model.predict({"scenario": "test"})
        assert result == {"prediction": "test_prediction"}
    
    def test_world_model_state_summary(self, test_world_model):
        """Test WorldModel state summary."""
        summary = test_world_model.get_state_summary()
        assert "observations" in summary
        assert summary["observations"] == 0
    
    def test_world_model_reset(self, test_world_model):
        """Test WorldModel reset functionality."""
        # Add some test data
        observation = ProcessedObservation(
            data={"test": "data"},
            timestamp=1234567890.0,
            source="test",
            confidence=0.9
        )
        test_world_model.update(observation)
        
        # Reset and verify
        test_world_model.reset()
        assert test_world_model.get_state_summary()["observations"] == 0
    
    def test_world_model_info(self, test_world_model):
        """Test WorldModel info retrieval."""
        info = test_world_model.get_model_info()
        
        assert "model_id" in info
        assert "model_type" in info
        assert "created_at" in info
        assert "last_updated" in info
        assert "registered_agents" in info
        assert "collaboration_events" in info


class TestMentalStateClass:
    """Test the MentalState class."""
    
    def test_mental_state_initialization(self, test_world_model):
        """Test MentalState initialization."""
        mental_state = MentalState(test_world_model)
        
        assert mental_state.world_model == test_world_model
        assert mental_state.goals == []
        assert mental_state.emotions == {}
        assert mental_state.memory == {}
        assert mental_state.context == {}
    
    def test_mental_state_world_model_update(self, test_world_model, sample_processed_observation):
        """Test MentalState world model update."""
        mental_state = MentalState(test_world_model)
        mental_state.update_world_model(sample_processed_observation)
        
        # Verify the world model was updated
        assert test_world_model.knowledge_base[sample_processed_observation.source] == sample_processed_observation.data
    
    def test_mental_state_world_model_query(self, test_world_model):
        """Test MentalState world model query."""
        mental_state = MentalState(test_world_model)
        test_world_model.knowledge_base["test_key"] = "test_value"
        
        result = mental_state.query_world_model("test_key")
        assert result == "test_value"
    
    def test_mental_state_prediction(self, test_world_model):
        """Test MentalState prediction functionality."""
        mental_state = MentalState(test_world_model)
        scenario = {"condition": "hypertension"}
        
        prediction = mental_state.predict_outcome(scenario)
        assert "prediction" in prediction
        assert "confidence" in prediction
    
    def test_mental_state_goals(self, test_world_model):
        """Test MentalState goal management."""
        mental_state = MentalState(test_world_model)
        
        # Add goals
        mental_state.add_goal("diagnose_condition")
        mental_state.add_goal("provide_treatment")
        assert len(mental_state.goals) == 2
        assert "diagnose_condition" in mental_state.goals
        
        # Remove goal
        mental_state.remove_goal("diagnose_condition")
        assert len(mental_state.goals) == 1
        assert "diagnose_condition" not in mental_state.goals
        
        # Try to add duplicate goal
        mental_state.add_goal("provide_treatment")
        assert len(mental_state.goals) == 1  # Should not add duplicate
    
    def test_mental_state_emotions(self, test_world_model):
        """Test MentalState emotion management."""
        mental_state = MentalState(test_world_model)
        
        # Set emotions
        mental_state.set_emotion("confidence", 0.8)
        mental_state.set_emotion("concern", 0.3)
        
        assert mental_state.emotions["confidence"] == 0.8
        assert mental_state.emotions["concern"] == 0.3
        
        # Test bounds enforcement
        mental_state.set_emotion("excitement", 1.5)  # Should be clamped to 1.0
        mental_state.set_emotion("worry", -0.5)      # Should be clamped to 0.0
        
        assert mental_state.emotions["excitement"] == 1.0
        assert mental_state.emotions["worry"] == 0.0
    
    def test_mental_state_summary(self, test_world_model):
        """Test MentalState summary generation."""
        mental_state = MentalState(test_world_model)
        mental_state.add_goal("test_goal")
        mental_state.set_emotion("confidence", 0.7)
        mental_state.memory["test_memory"] = "value"
        mental_state.context["test_context"] = "value"
        
        summary = mental_state.get_state_summary()
        
        assert "world_model" in summary
        assert "goals" in summary
        assert "emotions" in summary
        assert "memory_keys" in summary
        assert "context_keys" in summary
        
        assert summary["goals"] == ["test_goal"]
        assert summary["emotions"] == {"confidence": 0.7}
        assert summary["memory_keys"] == ["test_memory"]
        assert summary["context_keys"] == ["test_context"]


class TestMedicalAgentAbstract:
    """Test the abstract MedicalAgent class."""
    
    def test_cannot_instantiate_abstract_medical_agent(self):
        """Test that MedicalAgent cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MedicalAgent("test", "test", [], Mock())
    
    def test_medical_agent_has_required_abstract_methods(self):
        """Test that MedicalAgent has all required abstract methods."""
        abstract_methods = MedicalAgent.__abstractmethods__
        expected_methods = {"perceive", "learn", "reason", "execute", "build_agent_card"}
        assert abstract_methods == expected_methods


class TestMedicalAgentConcrete:
    """Test concrete MedicalAgent implementations."""
    
    def test_medical_agent_initialization(self, test_medical_agent, test_world_model):
        """Test MedicalAgent initialization."""
        assert test_medical_agent.agent_id == "test-agent"
        assert test_medical_agent.agent_type == "test"
        assert test_medical_agent.capabilities == ["testing", "validation"]
        assert test_medical_agent.mental_state.world_model == test_world_model
        assert test_medical_agent.discovery_service is None
        assert test_medical_agent._clients == {}
    
    @pytest.mark.asyncio
    async def test_medical_agent_perceive(self, test_medical_agent):
        """Test MedicalAgent perception functionality."""
        observation_data = {"symptom": "headache", "severity": "moderate"}
        
        result = await test_medical_agent.perceive(observation_data)
        
        assert isinstance(result, ProcessedObservation)
        assert result.data == observation_data
        assert result.source == "test_input"
        assert result.confidence == 0.9
        assert observation_data in test_medical_agent.perception_calls
    
    @pytest.mark.asyncio
    async def test_medical_agent_learn(self, test_medical_agent, sample_processed_observation):
        """Test MedicalAgent learning functionality."""
        initial_state = test_medical_agent.mental_state
        
        updated_state = await test_medical_agent.learn(initial_state, sample_processed_observation)
        
        assert updated_state == initial_state  # Should return the same state object
        assert (initial_state, sample_processed_observation) in test_medical_agent.learning_calls
    
    @pytest.mark.asyncio
    async def test_medical_agent_reason(self, test_medical_agent):
        """Test MedicalAgent reasoning functionality."""
        state = test_medical_agent.mental_state
        
        action = await test_medical_agent.reason(state)
        
        assert isinstance(action, Action)
        assert action.action_type == "test_action"
        assert action.parameters == {"test": "data"}
        assert action.priority == 1
        assert state in test_medical_agent.reasoning_calls
    
    @pytest.mark.asyncio
    async def test_medical_agent_execute(self, test_medical_agent, sample_action):
        """Test MedicalAgent execution functionality."""
        result = await test_medical_agent.execute(sample_action)
        
        assert isinstance(result, ActionResult)
        assert result.success is True
        assert result.data == {"executed": sample_action.action_type}
        assert result.metadata == sample_action.parameters
        assert sample_action in test_medical_agent.execution_calls
    
    @pytest.mark.asyncio
    async def test_medical_agent_process_request(self, test_medical_agent):
        """Test MedicalAgent full request processing pipeline."""
        mock_request = Mock()
        
        response = await test_medical_agent.process_request(mock_request)
        
        # Verify PCE cycle was executed
        assert len(test_medical_agent.perception_calls) == 1
        assert len(test_medical_agent.learning_calls) == 1
        assert len(test_medical_agent.reasoning_calls) == 1
        assert len(test_medical_agent.execution_calls) == 1
        
        # Verify response structure (mocked since we can't import A2A types)
        assert response is not None
    
    def test_medical_agent_client_management(self, test_medical_agent, mock_a2a_client):
        """Test MedicalAgent A2A client management."""
        client_id = "test-client"
        
        # Add client
        test_medical_agent.add_client(client_id, mock_a2a_client)
        assert test_medical_agent.get_client(client_id) == mock_a2a_client
        
        # Remove client
        test_medical_agent.remove_client(client_id)
        assert test_medical_agent.get_client(client_id) is None
    
    def test_medical_agent_info(self, test_medical_agent):
        """Test MedicalAgent info retrieval."""
        info = test_medical_agent.get_agent_info()
        
        assert "agent_id" in info
        assert "agent_type" in info
        assert "capabilities" in info
        assert "mental_state" in info
        assert "connected_clients" in info
        assert "is_active" in info
        assert "agent_card" in info
        
        assert info["agent_id"] == "test-agent"
        assert info["agent_type"] == "test"
        assert info["capabilities"] == ["testing", "validation"]


class TestCognitiveModuleAbstract:
    """Test the abstract CognitiveModule class."""
    
    def test_cannot_instantiate_abstract_cognitive_module(self):
        """Test that CognitiveModule cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CognitiveModule("test-module")
    
    def test_cognitive_module_has_required_abstract_methods(self):
        """Test that CognitiveModule has required abstract methods."""
        abstract_methods = CognitiveModule.__abstractmethods__
        expected_methods = {"process", "configure", "get_capabilities"}
        assert abstract_methods == expected_methods


class TestCognitiveModuleConcrete:
    """Test concrete CognitiveModule implementations."""
    
    def test_cognitive_module_initialization(self):
        """Test CognitiveModule initialization."""
        
        class TestCognitiveModule(CognitiveModule):
            async def process(self, input_data, context):
                return input_data
            
            def configure(self, config):
                self.config = config
            
            def get_capabilities(self):
                return ["test_capability"]
        
        module = TestCognitiveModule("test-module")
        
        assert module.module_id == "test-module"
        assert module.is_active_flag is False
        assert module.config == {}
        assert module.dependencies == []
        assert module.a2a_enabled is False
    
    def test_cognitive_module_activation(self):
        """Test CognitiveModule activation/deactivation."""
        
        class TestCognitiveModule(CognitiveModule):
            async def process(self, input_data, context):
                return input_data
            
            def configure(self, config):
                self.config = config
            
            def get_capabilities(self):
                return ["test_capability"]
        
        module = TestCognitiveModule("test-module")
        
        # Test initial state
        assert module.is_active() is False
        
        # Test activation
        module.activate()
        assert module.is_active() is True
        
        # Test deactivation
        module.deactivate()
        assert module.is_active() is False
    
    def test_cognitive_module_info(self):
        """Test CognitiveModule info retrieval."""
        
        class TestCognitiveModule(CognitiveModule):
            async def process(self, input_data, context):
                return input_data
            
            def configure(self, config):
                self.config = config
            
            def get_capabilities(self):
                return ["test_capability"]
        
        module = TestCognitiveModule("test-module")
        module.configure({"setting": "value"})
        
        info = module.get_module_info()
        
        assert "module_id" in info
        assert "module_type" in info
        assert "is_active" in info
        assert "capabilities" in info
        assert "dependencies" in info
        assert "a2a_enabled" in info
        assert "config_keys" in info
        
        assert info["module_id"] == "test-module"
        assert info["module_type"] == "TestCognitiveModule"
        assert info["is_active"] is False
        assert info["capabilities"] == ["test_capability"]
        assert info["dependencies"] == []
        assert info["a2a_enabled"] is False
        assert info["config_keys"] == ["setting"]


class TestDataClasses:
    """Test data classes used in the agent framework."""
    
    def test_processed_observation(self):
        """Test ProcessedObservation data class."""
        observation = ProcessedObservation(
            data={"test": "data"},
            timestamp=123456789.0,
            source="test_source",
            confidence=0.8
        )
        
        assert observation.data == {"test": "data"}
        assert observation.timestamp == 123456789.0
        assert observation.source == "test_source"
        assert observation.confidence == 0.8
    
    def test_action(self):
        """Test Action data class."""
        action = Action(
            action_type="test_action",
            parameters={"param": "value"},
            priority=2,
            metadata={"meta": "data"}
        )
        
        assert action.action_type == "test_action"
        assert action.parameters == {"param": "value"}
        assert action.priority == 2
        assert action.metadata == {"meta": "data"}
    
    def test_action_result(self):
        """Test ActionResult data class."""
        result = ActionResult(
            success=True,
            data={"result": "data"},
            error=None,
            metadata={"execution_time": 0.5}
        )
        
        assert result.success is True
        assert result.data == {"result": "data"}
        assert result.error is None
        assert result.metadata == {"execution_time": 0.5}
        
        # Test failure case
        failure = ActionResult(
            success=False,
            error="Test error",
            metadata={}
        )
        
        assert failure.success is False
        assert failure.error == "Test error"
        assert failure.data is None 
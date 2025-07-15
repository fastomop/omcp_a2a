"""
Comprehensive test suite for Ollama integration.

Tests all aspects of the Ollama integration including:
- OllamaReasoningMixin functionality
- ActionType enum and ParsedAction dataclass
- OllamaActionParser parsing methods
- JSON and XML parsing
- Action conversion
- Error handling and edge cases
"""

import pytest
import json
import re
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from a2a_medical.integrations.ollama import (
    OllamaReasoningMixin, ActionType, ParsedAction, OllamaActionParser
)
from a2a_medical.base.agent import Action, MedicalAgent, WorldModel, ProcessedObservation


class TestActionType:
    """Test ActionType enum."""
    
    def test_action_type_values(self):
        """Test ActionType enum values."""
        assert ActionType.TOOL_CALL == "tool_call"
        assert ActionType.A2A_MESSAGE == "a2a_message"
        assert ActionType.DIRECT_RESPONSE == "direct_response"
        assert ActionType.REQUEST_INFO == "request_info"
    
    def test_action_type_membership(self):
        """Test ActionType enum membership."""
        assert ActionType.TOOL_CALL in ActionType
        assert ActionType.A2A_MESSAGE in ActionType
        assert ActionType.DIRECT_RESPONSE in ActionType
        assert ActionType.REQUEST_INFO in ActionType


class TestParsedAction:
    """Test ParsedAction dataclass."""
    
    def test_parsed_action_creation(self):
        """Test creating a ParsedAction instance."""
        action = ParsedAction(
            action_type=ActionType.TOOL_CALL,
            target="test:tool",
            parameters={"param1": "value1"},
            reasoning="Testing tool call",
            confidence=0.8
        )
        
        assert action.action_type == ActionType.TOOL_CALL
        assert action.target == "test:tool"
        assert action.parameters == {"param1": "value1"}
        assert action.reasoning == "Testing tool call"
        assert action.confidence == 0.8
    
    def test_parsed_action_defaults(self):
        """Test ParsedAction with default values."""
        action = ParsedAction(action_type=ActionType.DIRECT_RESPONSE)
        
        assert action.action_type == ActionType.DIRECT_RESPONSE
        assert action.target is None
        assert action.parameters is None
        assert action.reasoning is None
        assert action.confidence == 0.5
    
    def test_parsed_action_equality(self):
        """Test ParsedAction equality comparison."""
        action1 = ParsedAction(
            action_type=ActionType.TOOL_CALL,
            target="test:tool",
            confidence=0.8
        )
        action2 = ParsedAction(
            action_type=ActionType.TOOL_CALL,
            target="test:tool",
            confidence=0.8
        )
        action3 = ParsedAction(
            action_type=ActionType.A2A_MESSAGE,
            target="test:tool",
            confidence=0.8
        )
        
        assert action1 == action2
        assert action1 != action3


class TestOllamaReasoningMixin:
    """Test OllamaReasoningMixin functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock agent class that uses the mixin
        class TestAgent(OllamaReasoningMixin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.agent_id = "test-agent"
        
        self.agent_class = TestAgent
    
    def test_mixin_initialization_defaults(self):
        """Test mixin initialization with default values."""
        agent = self.agent_class()
        
        assert agent.ollama_model == "llama3.1:8b"
        assert agent.ollama_temperature == 0.1
        assert hasattr(agent, 'mcp_manager')
    
    def test_mixin_initialization_custom_values(self):
        """Test mixin initialization with custom values."""
        agent = self.agent_class(
            model_name="llama3.1:70b",
            ollama_temperature=0.5
        )
        
        assert agent.ollama_model == "llama3.1:70b"
        assert agent.ollama_temperature == 0.5
    
    @patch('ollama.show')
    @patch('ollama.pull')
    def test_ensure_ollama_model_exists(self, mock_pull, mock_show):
        """Test ensuring Ollama model when it exists."""
        mock_show.return_value = {"name": "llama3.1:8b"}
        
        agent = self.agent_class()
        agent._ensure_ollama_model()
        
        mock_show.assert_called_once_with("llama3.1:8b")
        mock_pull.assert_not_called()
    
    @patch('ollama.show')
    @patch('ollama.pull')
    def test_ensure_ollama_model_download(self, mock_pull, mock_show):
        """Test ensuring Ollama model when it needs to be downloaded."""
        mock_show.side_effect = Exception("Model not found")
        
        agent = self.agent_class()
        agent._ensure_ollama_model()
        
        mock_show.assert_called_once_with("llama3.1:8b")
        mock_pull.assert_called_once_with("llama3.1:8b")
    
    @patch('ollama.chat')
    async def test_ollama_reason_basic(self, mock_chat):
        """Test basic Ollama reasoning."""
        mock_chat.return_value = {
            "message": {"content": '{"response": "test response"}'}
        }
        
        agent = self.agent_class()
        result = await agent.ollama_reason("Test prompt")
        
        assert result == {"response": "test response"}
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args
        assert call_args[1]["model"] == "llama3.1:8b"
        assert call_args[1]["options"]["temperature"] == 0.1
    
    @patch('ollama.chat')
    async def test_ollama_reason_with_system_prompt(self, mock_chat):
        """Test Ollama reasoning with system prompt."""
        mock_chat.return_value = {
            "message": {"content": '{"response": "test response"}'}
        }
        
        agent = self.agent_class()
        result = await agent.ollama_reason(
            "Test prompt",
            system_prompt="You are a medical assistant."
        )
        
        assert result == {"response": "test response"}
        call_args = mock_chat.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a medical assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test prompt"
    
    @patch('ollama.chat')
    async def test_ollama_reason_without_tools(self, mock_chat):
        """Test Ollama reasoning without including tools."""
        mock_chat.return_value = {
            "message": {"content": '{"response": "test response"}'}
        }
        
        agent = self.agent_class()
        result = await agent.ollama_reason("Test prompt", include_tools=False)
        
        assert result == {"response": "test response"}
        call_args = mock_chat.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test prompt"
    
    @patch('ollama.chat')
    async def test_ollama_reason_with_tools(self, mock_chat):
        """Test Ollama reasoning with tools included."""
        mock_chat.return_value = {
            "message": {"content": '{"response": "test response"}'}
        }
        
        # Mock MCP manager
        agent = self.agent_class()
        agent.mcp_manager.get_available_tools = AsyncMock(return_value="Available tools")
        
        result = await agent.ollama_reason("Test prompt", include_tools=True)
        
        assert result == {"response": "test response"}
        call_args = mock_chat.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "AVAILABLE TOOLS:" in messages[0]["content"]
        assert "Test prompt" in messages[0]["content"]
    
    @patch('ollama.chat')
    async def test_ollama_reason_json_parse_error(self, mock_chat):
        """Test Ollama reasoning with JSON parse error."""
        mock_chat.return_value = {
            "message": {"content": "Plain text response"}
        }
        
        agent = self.agent_class()
        result = await agent.ollama_reason("Test prompt")
        
        assert result == {"response": "Plain text response"}
    
    @patch('ollama.chat')
    async def test_ollama_reason_exception(self, mock_chat):
        """Test Ollama reasoning with exception."""
        mock_chat.side_effect = Exception("Ollama error")
        
        agent = self.agent_class()
        result = await agent.ollama_reason("Test prompt")
        
        assert "error" in result
        assert "Ollama reasoning failed" in result["error"]


class TestOllamaActionParser:
    """Test OllamaActionParser functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = OllamaActionParser()
    
    def test_action_format_prompt(self):
        """Test ACTION_FORMAT_PROMPT is defined."""
        assert hasattr(OllamaActionParser, 'ACTION_FORMAT_PROMPT')
        assert isinstance(OllamaActionParser.ACTION_FORMAT_PROMPT, str)
        assert "tool_call" in OllamaActionParser.ACTION_FORMAT_PROMPT
        assert "a2a_message" in OllamaActionParser.ACTION_FORMAT_PROMPT
    
    def test_parse_ollama_output_json_success(self):
        """Test parsing JSON output successfully."""
        json_output = json.dumps({
            "action_type": "tool_call",
            "tool_id": "test:tool",
            "parameters": {"param1": "value1"},
            "reasoning": "Testing tool",
            "confidence": 0.8
        })
        
        actions = OllamaActionParser.parse_ollama_output(json_output)
        
        assert len(actions) == 1
        action = actions[0]
        assert action.action_type == ActionType.TOOL_CALL
        assert action.target == "test:tool"
        assert action.parameters == {"param1": "value1"}
        assert action.reasoning == "Testing tool"
        assert action.confidence == 0.8
    
    def test_parse_ollama_output_json_invalid(self):
        """Test parsing invalid JSON output."""
        invalid_json = "{invalid json"
        
        actions = OllamaActionParser.parse_ollama_output(invalid_json)
        
        # Should fall back to XML parsing
        assert len(actions) == 0
    
    def test_parse_ollama_output_xml_tool_call(self):
        """Test parsing XML tool call."""
        xml_output = """
        <tool_call>
        tool_id: test:tool
        parameters: {"param1": "value1"}
        reasoning: Testing tool call
        </tool_call>
        """
        
        actions = OllamaActionParser.parse_ollama_output(xml_output)
        
        assert len(actions) == 1
        action = actions[0]
        assert action.action_type == ActionType.TOOL_CALL
        assert action.target == "test:tool"
        assert action.parameters == {"param1": "value1"}
        assert action.reasoning == "Testing tool call"
        assert action.confidence == 0.8
    
    def test_parse_ollama_output_xml_a2a_message(self):
        """Test parsing XML A2A message."""
        xml_output = """
        <a2a_message>
        agent_id: cardiologist
        message_type: consultation
        content: Patient needs cardiac evaluation
        reasoning: Symptoms suggest cardiac issues
        </a2a_message>
        """
        
        actions = OllamaActionParser.parse_ollama_output(xml_output)
        
        assert len(actions) == 1
        action = actions[0]
        assert action.action_type == ActionType.A2A_MESSAGE
        assert action.target == "cardiologist"
        assert action.parameters["message_type"] == "consultation"
        assert action.parameters["content"] == "Patient needs cardiac evaluation"
        assert action.reasoning == "Symptoms suggest cardiac issues"
        assert action.confidence == 0.8
    
    def test_parse_ollama_output_xml_direct_response(self):
        """Test parsing XML direct response."""
        xml_output = """
        <direct_response>
        assessment: Patient has mild symptoms
        recommendations: ["rest", "hydration"]
        confidence: 0.7
        follow_up_needed: false
        </direct_response>
        """
        
        actions = OllamaActionParser.parse_ollama_output(xml_output)
        
        assert len(actions) == 1
        action = actions[0]
        assert action.action_type == ActionType.DIRECT_RESPONSE
        assert action.parameters["text"] == xml_output.strip()
        assert action.confidence == 0.5
    
    def test_parse_ollama_output_xml_request_info(self):
        """Test parsing XML request info."""
        xml_output = """
        <request_info>
        needed_info: ["blood_pressure", "heart_rate"]
        from: patient
        reasoning: Need vital signs for assessment
        </request_info>
        """
        
        actions = OllamaActionParser.parse_ollama_output(xml_output)
        
        assert len(actions) == 1
        action = actions[0]
        assert action.action_type == ActionType.REQUEST_INFO
        assert action.parameters["text"] == xml_output.strip()
        assert action.confidence == 0.5
    
    def test_parse_ollama_output_multiple_actions(self):
        """Test parsing multiple actions from XML."""
        xml_output = """
        <tool_call>
        tool_id: lab:blood_test
        parameters: {"test_type": "cbc"}
        reasoning: Need blood work
        </tool_call>
        <a2a_message>
        agent_id: specialist
        message_type: referral
        content: Refer to specialist
        reasoning: Complex case
        </a2a_message>
        """
        
        actions = OllamaActionParser.parse_ollama_output(xml_output)
        
        assert len(actions) == 2
        assert actions[0].action_type == ActionType.TOOL_CALL
        assert actions[1].action_type == ActionType.A2A_MESSAGE
    
    def test_parse_ollama_output_plain_text(self):
        """Test parsing plain text as direct response."""
        plain_text = "Patient appears healthy. No immediate concerns."
        
        actions = OllamaActionParser.parse_ollama_output(plain_text)
        
        assert len(actions) == 1
        action = actions[0]
        assert action.action_type == ActionType.DIRECT_RESPONSE
        assert action.parameters["text"] == plain_text.strip()
        assert action.confidence == 0.5
    
    def test_parse_ollama_output_empty(self):
        """Test parsing empty output."""
        actions = OllamaActionParser.parse_ollama_output("")
        
        assert len(actions) == 0
    
    def test_parse_xml_actions_tool_call_invalid(self):
        """Test parsing invalid tool call XML."""
        invalid_xml = """
        <tool_call>
        tool_id: 
        parameters: invalid json
        </tool_call>
        """
        
        actions = OllamaActionParser._parse_xml_actions(invalid_xml)
        
        assert len(actions) == 0
    
    def test_parse_xml_actions_a2a_message_invalid(self):
        """Test parsing invalid A2A message XML."""
        invalid_xml = """
        <a2a_message>
        agent_id: 
        content: 
        </a2a_message>
        """
        
        actions = OllamaActionParser._parse_xml_actions(invalid_xml)
        
        assert len(actions) == 0
    
    def test_parse_json_action_tool_call(self):
        """Test parsing JSON tool call action."""
        json_data = {
            "action_type": "tool_call",
            "tool_id": "test:tool",
            "parameters": {"param1": "value1"},
            "reasoning": "Testing",
            "confidence": 0.8
        }
        
        action = OllamaActionParser._parse_json_action(json_data)
        
        assert action is not None
        assert action.action_type == ActionType.TOOL_CALL
        assert action.target == "test:tool"
        assert action.parameters == {"param1": "value1"}
        assert action.reasoning == "Testing"
        assert action.confidence == 0.8
    
    def test_parse_json_action_a2a_message(self):
        """Test parsing JSON A2A message action."""
        json_data = {
            "action_type": "a2a_message",
            "agent_id": "specialist",
            "message_type": "consultation",
            "content": "Need consultation",
            "reasoning": "Complex case",
            "confidence": 0.7
        }
        
        action = OllamaActionParser._parse_json_action(json_data)
        
        assert action is not None
        assert action.action_type == ActionType.A2A_MESSAGE
        assert action.target == "specialist"
        assert action.parameters["message_type"] == "consultation"
        assert action.parameters["content"] == "Need consultation"
        assert action.reasoning == "Complex case"
        assert action.confidence == 0.7
    
    def test_parse_json_action_direct_response(self):
        """Test parsing JSON direct response action."""
        json_data = {
            "action_type": "direct_response",
            "parameters": {"assessment": "Healthy"},
            "reasoning": "No symptoms",
            "confidence": 0.9
        }
        
        action = OllamaActionParser._parse_json_action(json_data)
        
        assert action is not None
        assert action.action_type == ActionType.DIRECT_RESPONSE
        assert action.parameters == {"assessment": "Healthy"}
        assert action.reasoning == "No symptoms"
        assert action.confidence == 0.9
    
    def test_parse_json_action_request_info(self):
        """Test parsing JSON request info action."""
        json_data = {
            "action_type": "request_info",
            "parameters": {"needed": ["vitals"]},
            "reasoning": "Need more data",
            "confidence": 0.6
        }
        
        action = OllamaActionParser._parse_json_action(json_data)
        
        assert action is not None
        assert action.action_type == ActionType.REQUEST_INFO
        assert action.parameters == {"needed": ["vitals"]}
        assert action.reasoning == "Need more data"
        assert action.confidence == 0.6
    
    def test_parse_json_action_unknown_type(self):
        """Test parsing JSON with unknown action type."""
        json_data = {
            "action_type": "unknown_action",
            "parameters": {}
        }
        
        action = OllamaActionParser._parse_json_action(json_data)
        
        assert action is None
    
    def test_convert_to_agent_action_tool_call(self):
        """Test converting tool call to agent action."""
        parsed_action = ParsedAction(
            action_type=ActionType.TOOL_CALL,
            target="test:tool",
            parameters={"param1": "value1"},
            reasoning="Testing tool",
            confidence=0.8
        )
        
        action = OllamaActionParser.convert_to_agent_action(parsed_action)
        
        assert action.action_type == "execute_mcp_tool"
        assert action.parameters["tool_id"] == "test:tool"
        assert action.parameters["tool_parameters"] == {"param1": "value1"}
        assert action.parameters["reasoning"] == "Testing tool"
        assert action.priority == 1
        assert action.metadata["confidence"] == 0.8
    
    def test_convert_to_agent_action_a2a_message(self):
        """Test converting A2A message to agent action."""
        parsed_action = ParsedAction(
            action_type=ActionType.A2A_MESSAGE,
            target="specialist",
            parameters={
                "message_type": "consultation",
                "content": "Need consultation"
            },
            reasoning="Complex case",
            confidence=0.7
        )
        
        action = OllamaActionParser.convert_to_agent_action(parsed_action)
        
        assert action.action_type == "send_a2a_message"
        assert action.parameters["target_agent"] == "specialist"
        assert action.parameters["message"] == "Need consultation"
        assert action.parameters["message_type"] == "consultation"
        assert action.parameters["reasoning"] == "Complex case"
        assert action.priority == 2
        assert action.metadata["confidence"] == 0.7
    
    def test_convert_to_agent_action_direct_response(self):
        """Test converting direct response to agent action."""
        parsed_action = ParsedAction(
            action_type=ActionType.DIRECT_RESPONSE,
            parameters={"text": "Patient is healthy"},
            confidence=0.9
        )
        
        action = OllamaActionParser.convert_to_agent_action(parsed_action)
        
        assert action.action_type == "provide_medical_response"
        assert action.parameters == {"text": "Patient is healthy"}
        assert action.priority == 3
        assert action.metadata["confidence"] == 0.9
    
    def test_convert_to_agent_action_request_info(self):
        """Test converting request info to agent action."""
        parsed_action = ParsedAction(
            action_type=ActionType.REQUEST_INFO,
            parameters={"text": "Need vitals"},
            confidence=0.6
        )
        
        action = OllamaActionParser.convert_to_agent_action(parsed_action)
        
        assert action.action_type == "request_additional_info"
        assert action.parameters == {"text": "Need vitals"}
        assert action.priority == 4
        assert action.metadata["confidence"] == 0.6
    
    def test_convert_to_agent_action_with_context(self):
        """Test converting action with agent context."""
        parsed_action = ParsedAction(
            action_type=ActionType.TOOL_CALL,
            target="test:tool",
            parameters={"param1": "value1"},
            confidence=0.8
        )
        
        agent_context = {"agent_id": "test-agent", "specialty": "cardiology"}
        action = OllamaActionParser.convert_to_agent_action(parsed_action, agent_context)
        
        assert action.action_type == "execute_mcp_tool"
        assert action.parameters["tool_id"] == "test:tool"
        assert action.metadata["confidence"] == 0.8


class TestOllamaIntegrationWithMedicalAgent:
    """Test Ollama integration with medical agents."""
    
    def setup_method(self):
        """Set up test fixtures."""
        class TestWorldModel(WorldModel):
            def update(self, observation: ProcessedObservation) -> None:
                pass
            
            def query(self, query: str, context=None):
                return "test result"
            
            def predict(self, scenario):
                return {"prediction": "test"}
            
            def get_state_summary(self):
                return {"status": "active"}
            
            def reset(self) -> None:
                pass
        
        class TestMedicalAgent(MedicalAgent, OllamaReasoningMixin):
            def __init__(self, *args, **kwargs):
                MedicalAgent.__init__(self, *args, **kwargs)
                OllamaReasoningMixin.__init__(self, *args, **kwargs)
            
            async def perceive(self, observation):
                return ProcessedObservation(
                    data=observation,
                    timestamp=datetime.now().timestamp(),
                    source="test",
                    confidence=0.9
                )
            
            async def learn(self, state, observation):
                return state
            
            async def reason(self, state):
                # Use Ollama for reasoning
                result = await self.ollama_reason("Analyze patient state")
                return Action(
                    action_type="test_action",
                    parameters=result,
                    priority=1
                )
            
            async def execute(self, action):
                from a2a_medical.base.agent import ActionResult
                return ActionResult(
                    success=True,
                    data={"result": "test"},
                    error=None,
                    metadata={"action": action.action_type}
                )
            
            def build_agent_card(self):
                from a2a.types import AgentCard, AgentCapabilities
                return AgentCard(
                    name=self.agent_name,
                    description=self.agent_description,
                    url="https://test.local",
                    version=self.agent_version,
                    capabilities=AgentCapabilities(streaming=False),
                    skills=[],
                    defaultInputModes=["text"],
                    defaultOutputModes=["text"]
                )
        
        self.agent_class = TestMedicalAgent
    
    def test_medical_agent_with_ollama_integration(self):
        """Test medical agent with Ollama integration."""
        world_model = TestWorldModel()
        
        agent = self.agent_class(
            agent_id="ollama-test-agent",
            agent_type="test",
            capabilities=["diagnosis"],
            world_model=world_model,
            model_name="llama3.1:8b",
            ollama_temperature=0.2
        )
        
        assert agent.agent_id == "ollama-test-agent"
        assert agent.ollama_model == "llama3.1:8b"
        assert agent.ollama_temperature == 0.2
        assert hasattr(agent, 'mcp_manager')
    
    @patch('ollama.chat')
    async def test_medical_agent_ollama_reasoning(self, mock_chat):
        """Test medical agent using Ollama for reasoning."""
        mock_chat.return_value = {
            "message": {"content": '{"assessment": "healthy", "confidence": 0.8}'}
        }
        
        world_model = TestWorldModel()
        agent = self.agent_class(
            agent_id="ollama-test-agent",
            agent_type="test",
            capabilities=["diagnosis"],
            world_model=world_model
        )
        
        # Test reasoning
        result = await agent.ollama_reason("Analyze patient symptoms")
        
        assert result == {"assessment": "healthy", "confidence": 0.8}
        mock_chat.assert_called_once()
    
    async def test_medical_agent_action_parsing(self):
        """Test medical agent parsing Ollama actions."""
        world_model = TestWorldModel()
        agent = self.agent_class(
            agent_id="ollama-test-agent",
            agent_type="test",
            capabilities=["diagnosis"],
            world_model=world_model
        )
        
        # Test parsing tool call
        tool_call_xml = """
        <tool_call>
        tool_id: lab:blood_test
        parameters: {"test_type": "cbc"}
        reasoning: Need blood work for diagnosis
        </tool_call>
        """
        
        actions = OllamaActionParser.parse_ollama_output(tool_call_xml)
        
        assert len(actions) == 1
        action = actions[0]
        assert action.action_type == ActionType.TOOL_CALL
        assert action.target == "lab:blood_test"
        
        # Convert to agent action
        agent_action = OllamaActionParser.convert_to_agent_action(action)
        assert agent_action.action_type == "execute_mcp_tool"
        assert agent_action.parameters["tool_id"] == "lab:blood_test"


class TestOllamaErrorHandling:
    """Test Ollama error handling and edge cases."""
    
    def test_parse_tool_call_missing_fields(self):
        """Test parsing tool call with missing fields."""
        content = """
        tool_id: test:tool
        reasoning: Testing
        """
        
        action = OllamaActionParser._parse_tool_call(content)
        
        assert action is not None
        assert action.action_type == ActionType.TOOL_CALL
        assert action.target == "test:tool"
        assert action.parameters == {}
        assert action.reasoning == "Testing"
    
    def test_parse_tool_call_invalid_json(self):
        """Test parsing tool call with invalid JSON parameters."""
        content = """
        tool_id: test:tool
        parameters: {invalid json}
        reasoning: Testing
        """
        
        action = OllamaActionParser._parse_tool_call(content)
        
        assert action is not None
        assert action.action_type == ActionType.TOOL_CALL
        assert action.target == "test:tool"
        assert action.parameters == {}
    
    def test_parse_a2a_message_missing_fields(self):
        """Test parsing A2A message with missing fields."""
        content = """
        agent_id: specialist
        content: Need consultation
        reasoning: Complex case
        """
        
        action = OllamaActionParser._parse_a2a_message(content)
        
        assert action is not None
        assert action.action_type == ActionType.A2A_MESSAGE
        assert action.target == "specialist"
        assert action.parameters["content"] == "Need consultation"
        assert action.parameters["message_type"] == "consultation"
    
    def test_parse_a2a_message_missing_content(self):
        """Test parsing A2A message with missing content."""
        content = """
        agent_id: specialist
        message_type: consultation
        reasoning: Complex case
        """
        
        action = OllamaActionParser._parse_a2a_message(content)
        
        assert action is None
    
    def test_parse_direct_response_stub(self):
        """Test direct response parsing stub."""
        content = "Patient is healthy"
        
        action = OllamaActionParser._parse_direct_response(content)
        
        assert action is not None
        assert action.action_type == ActionType.DIRECT_RESPONSE
        assert action.parameters["text"] == "Patient is healthy"
        assert action.confidence == 0.5
    
    def test_parse_info_request_stub(self):
        """Test info request parsing stub."""
        content = "Need patient vitals"
        
        action = OllamaActionParser._parse_info_request(content)
        
        assert action is not None
        assert action.action_type == ActionType.REQUEST_INFO
        assert action.parameters["text"] == "Need patient vitals"
        assert action.confidence == 0.5


class TestOllamaPerformance:
    """Test Ollama performance characteristics."""
    
    def test_parse_large_xml_output(self):
        """Test parsing large XML output."""
        # Create large XML with multiple actions
        xml_parts = []
        for i in range(10):
            xml_parts.append(f"""
            <tool_call>
            tool_id: tool{i}
            parameters: {{"param{i}": "value{i}"}}
            reasoning: Testing tool {i}
            </tool_call>
            """)
        
        large_xml = "".join(xml_parts)
        
        actions = OllamaActionParser.parse_ollama_output(large_xml)
        
        assert len(actions) == 10
        for i, action in enumerate(actions):
            assert action.action_type == ActionType.TOOL_CALL
            assert action.target == f"tool{i}"
    
    def test_parse_mixed_content(self):
        """Test parsing mixed JSON and XML content."""
        mixed_content = """
        {"action_type": "tool_call", "tool_id": "json_tool"}
        <tool_call>
        tool_id: xml_tool
        parameters: {"param": "value"}
        </tool_call>
        """
        
        actions = OllamaActionParser.parse_ollama_output(mixed_content)
        
        # Should parse JSON first and return only that
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.TOOL_CALL
        assert actions[0].target == "json_tool"


# Integration tests
class TestOllamaIntegration:
    """Integration tests for Ollama functionality."""
    
    async def test_full_ollama_workflow(self):
        """Test complete Ollama workflow."""
        # Test parsing and conversion workflow
        xml_output = """
        <tool_call>
        tool_id: lab:blood_test
        parameters: {"test_type": "cbc"}
        reasoning: Need blood work for diagnosis
        </tool_call>
        """
        
        # Parse output
        actions = OllamaActionParser.parse_ollama_output(xml_output)
        assert len(actions) == 1
        
        # Convert to agent action
        agent_action = OllamaActionParser.convert_to_agent_action(actions[0])
        assert agent_action.action_type == "execute_mcp_tool"
        assert agent_action.parameters["tool_id"] == "lab:blood_test"
    
    async def test_ollama_medical_agent_integration(self):
        """Test Ollama integration with medical agent."""
        class TestWorldModel(WorldModel):
            def update(self, observation: ProcessedObservation) -> None:
                pass
            
            def query(self, query: str, context=None):
                return "test result"
            
            def predict(self, scenario):
                return {"prediction": "test"}
            
            def get_state_summary(self):
                return {"status": "active"}
            
            def reset(self) -> None:
                pass
        
        class TestMedicalAgent(MedicalAgent, OllamaReasoningMixin):
            def __init__(self, *args, **kwargs):
                MedicalAgent.__init__(self, *args, **kwargs)
                OllamaReasoningMixin.__init__(self, *args, **kwargs)
            
            async def perceive(self, observation):
                return ProcessedObservation(
                    data=observation,
                    timestamp=datetime.now().timestamp(),
                    source="test",
                    confidence=0.9
                )
            
            async def learn(self, state, observation):
                return state
            
            async def reason(self, state):
                return Action(
                    action_type="test_action",
                    parameters={"test": "value"},
                    priority=1
                )
            
            async def execute(self, action):
                from a2a_medical.base.agent import ActionResult
                return ActionResult(
                    success=True,
                    data={"result": "test"},
                    error=None,
                    metadata={"action": action.action_type}
                )
            
            def build_agent_card(self):
                from a2a.types import AgentCard, AgentCapabilities
                return AgentCard(
                    name=self.agent_name,
                    description=self.agent_description,
                    url="https://test.local",
                    version=self.agent_version,
                    capabilities=AgentCapabilities(streaming=False),
                    skills=[],
                    defaultInputModes=["text"],
                    defaultOutputModes=["text"]
                )
        
        # Create agent with Ollama integration
        world_model = TestWorldModel()
        agent = TestMedicalAgent(
            agent_id="integration-test",
            agent_type="test",
            capabilities=["diagnosis"],
            world_model=world_model,
            model_name="llama3.1:8b"
        )
        
        # Test basic functionality
        assert agent.agent_id == "integration-test"
        assert agent.ollama_model == "llama3.1:8b"
        assert hasattr(agent, 'mcp_manager')
        
        # Test agent can process observations
        observation = await agent.perceive({"test": "data"})
        assert observation.source == "test"
        assert observation.confidence == 0.9 
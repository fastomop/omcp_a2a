from typing import Optional, Dict, Any, List, Union
import ollama
from abc import ABC
from a2a_medical.integrations.mcp import MCPManager
from enum import Enum
from dataclasses import dataclass
import json
import re
from a2a_medical.base.agent import Action

class OllamaReasoningMixin:
    """Mixin to add Ollama LLM reasoning capabilities to any medical agent."""

    def __init__(self, *args, model_name: str = "llama3.1:8b", ollama_temperature: float = 0.1, **kwargs):
        
        self.ollama_model = model_name
        self.ollama_temperature = ollama_temperature
        self._ensure_ollama_model()
        self.mcp_manager = MCPManager([])
        super().__init__(*args, **kwargs)

    def _ensure_ollama_model(self):
        """Ensure the Ollama model is available."""
        try:
            ollama.show(self.ollama_model)
        except:
            print(f"📥 Downloading {self.ollama_model}...")
            ollama.pull(self.ollama_model)

    async def ollama_reason(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        include_tools: bool = True
    ) -> Dict[str, Any]:
        """
        Use Ollama for reasoning. Can be called from your custom reason() method.
          
        Returns structured response that you can parse in your agent.
        """
        if include_tools and hasattr(self, 'mcp_manager'):
            # Include available tools in context
            tools_context = await self.mcp_manager.get_available_tools()
            prompt = f"AVAILABLE TOOLS:\n{tools_context}\n\n{prompt}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=messages,
                options={
                    "temperature": self.ollama_temperature,
                    "top_p": 0.9,
                    "format": "json"  # Request JSON response
                }
            )

            # Try to parse as JSON, fallback to text
            content = response['message']['content']
            try:
                import json
                return json.loads(content)
            except:
                return {"response": content}

        except Exception as e:
            return {"error": f"Ollama reasoning failed: {str(e)}"}


class ActionType(Enum):
     TOOL_CALL = "tool_call"
     A2A_QUERY = "a2a_query"
     A2A_RESPONSE = "a2a_response"
     A2A_NOTIFICATION = "a2a_notification"
     A2A_EVENT = "a2a_event"

@dataclass
class ParsedAction:
    """Parsed action from LLM reasoning."""
    action_type: ActionType
    target: Optional[str] = None  # tool_id or agent_id
    parameters: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    confidence: float = 0.5

class OllamaActionParser:
    """
    Parses Ollama reasoning output and produces executable actions.
    Handles tool calls, A2A messages, and direct responses.
    """

      # Prompts that guide Ollama to produce parseable outputs
    ACTION_FORMAT_PROMPT = """
When you need to take an action, ALL outputs must be A2A messages. Use one of these formats:

1. To call a tool (results will be sent as A2A message):
<tool_call>
tool_id: {server}:{tool_name}
parameters: {
    "param1": "value1",
    "param2": "value2"
}
reasoning: Why you're using this tool
send_result_to: ["recipient_agent_id1", "recipient_agent_id2"]
</tool_call>

2. To send a QUERY message (asking for information):
<a2a_query>
recipient_ids: ["agent_id_1", "agent_id_2"]
query_content: {
    "question": "Your medical question",
    "context": {...},
    "urgency_level": 1-5
}
clinical_priority: 1-5
phi_present: true|false
reasoning: Why you're asking this
</a2a_query>

3. To send a RESPONSE message (answering a query):
<a2a_response>
recipient_ids: ["requesting_agent_id"]
in_reply_to: {original_message_id}
response_content: {
    "answer": "Your medical assessment or information",
    "confidence": 0.0-1.0,
    "evidence": [...],
    "recommendations": [...]
}
clinical_priority: 1-5
phi_present: true|false
</a2a_response>

4. To send a NOTIFICATION message (alerts/updates):
<a2a_notification>
recipient_ids: ["agent_id_1", "agent_id_2"]
notification_content: {
    "alert_type": "lab_result|status_update|clinical_alert",
    "message": "Notification details",
    "severity": "low|normal|high|urgent|critical"
}
priority: LOW|NORMAL|HIGH|URGENT|CRITICAL
requires_acknowledgment: true|false
</a2a_notification>

5. To send an EVENT message (medical events):
<a2a_event>
recipient_ids: ["agent_id_1", "agent_id_2"]
event_content: {
    "event_type": "admission|discharge|procedure|medication_administered",
    "description": "Event details",
    "timestamp": "ISO timestamp"
}
clinical_priority: 1-5
requires_audit: true|false
</a2a_event>

IMPORTANT: There are NO direct responses. All agent outputs must be proper A2A messages.
"""

    @staticmethod
    def parse_ollama_output(
        raw_output: str,
        available_tools: Optional[Dict[str, Any]] = None,
        available_agents: Optional[List[str]] = None
    ) -> List[ParsedAction]:
        """
        Parse Ollama output to extract structured actions.
        
        Args:
            raw_output: Raw text output from Ollama
            available_tools: Dict of available MCP tools for validation
            available_agents: List of available A2A agents for validation
              
        Returns:
            List of parsed actions (can be multiple)
        """
        actions = []

        # Try JSON parsing first (if Ollama was set to JSON mode)
        try:
            json_data = json.loads(raw_output)
            if action := OllamaActionParser._parse_json_action(
                json_data, available_tools, available_agents
            ):
                actions.append(action)
                return actions
        except json.JSONDecodeError:
            pass

        # Parse XML-style tags
        actions.extend(OllamaActionParser._parse_xml_actions(
            raw_output, available_tools, available_agents
        ))

        # If no structured actions found, treat as A2A notification
        if not actions and raw_output.strip():
            actions.append(ParsedAction(
                action_type=ActionType.A2A_NOTIFICATION,
                parameters={
                    "message_type": "NOTIFICATION",
                    "recipient_ids": [],  # Will need to be filled by agent context
                    "notification_content": {
                        "message": raw_output.strip(),
                        "alert_type": "unstructured_response"
                    },
                    "priority": "NORMAL"
                },
                confidence=0.5
            ))

        return actions

    @staticmethod
    def _parse_xml_actions(
        text: str,
        available_tools: Optional[Dict[str, Any]] = None,
        available_agents: Optional[List[str]] = None
    ) -> List[ParsedAction]:
        """Parse XML-style action tags from text."""
        actions = []

        # Parse tool calls
        tool_pattern = r'<tool_call>(.*?)</tool_call>'
        for match in re.finditer(tool_pattern, text, re.DOTALL):
            content = match.group(1)
            if action := OllamaActionParser._parse_tool_call(
                content, available_tools or {}
            ):
                actions.append(action)

        # Parse A2A QUERY messages
        query_pattern = r'<a2a_query>(.*?)</a2a_query>'
        for match in re.finditer(query_pattern, text, re.DOTALL):
            content = match.group(1)
            if action := OllamaActionParser._parse_a2a_query(content):
                actions.append(action)

        # Parse A2A RESPONSE messages
        response_pattern = r'<a2a_response>(.*?)</a2a_response>'
        for match in re.finditer(response_pattern, text, re.DOTALL):
            content = match.group(1)
            if action := OllamaActionParser._parse_a2a_response(content):
                actions.append(action)

        # Parse A2A NOTIFICATION messages
        notification_pattern = r'<a2a_notification>(.*?)</a2a_notification>'
        for match in re.finditer(notification_pattern, text, re.DOTALL):
            content = match.group(1)
            if action := OllamaActionParser._parse_a2a_notification(content):
                actions.append(action)

        # Parse A2A EVENT messages
        event_pattern = r'<a2a_event>(.*?)</a2a_event>'
        for match in re.finditer(event_pattern, text, re.DOTALL):
            content = match.group(1)
            if action := OllamaActionParser._parse_a2a_event(content):
                actions.append(action)

        return actions

    @staticmethod
    def _parse_tool_call(
        content: str,
        available_tools: Optional[Dict[str, Any]] = None
    ) -> Optional[ParsedAction]:
        """Parse a tool call action."""
        available_tools = available_tools or {}
        try:
            # Extract fields
            tool_id = re.search(r'tool_id:\s*([^\n]+)', content)
            params = re.search(r'parameters:\s*({[^}]+})', content, re.DOTALL)
            reasoning = re.search(r'reasoning:\s*([^\n]+)', content)
            send_to_match = re.search(r'send_result_to:\s*\[(.*?)\]', content, re.DOTALL)

            if not tool_id:
                return None

            tool_id = tool_id.group(1).strip()

            # Parse send_result_to recipients
            send_result_to = []
            if send_to_match:
                recipients_str = send_to_match.group(1)
                send_result_to = [r.strip().strip('"\'') for r in recipients_str.split(',') if r.strip()]

            parameters = {}
            if params:
                try:
                    parameters = json.loads(params.group(1))
                except:
                    pass

            return ParsedAction(
                action_type=ActionType.TOOL_CALL,
                target=tool_id,
                parameters={
                    "tool_parameters": parameters,
                    "send_result_to": send_result_to
                },
                reasoning=reasoning.group(1).strip() if reasoning else None,
                confidence=0.8
            )
        except Exception:
            return None

    @staticmethod
    def _parse_recipients(content: str) -> Optional[List[str]]:
        """Helper to parse recipient_ids list."""
        recipient_match = re.search(r'recipient_ids:\s*\[(.*?)\]', content, re.DOTALL)
        if not recipient_match:
            return None
        recipients_str = recipient_match.group(1)
        return [r.strip().strip('"\'') for r in recipients_str.split(',') if r.strip()]

    @staticmethod
    def _parse_json_content(content: str, field_name: str) -> Optional[Dict[str, Any]]:
        """Helper to parse JSON content fields."""
        pattern = rf'{field_name}:\s*(\{{.*?\}})'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _parse_a2a_query(content: str) -> Optional[ParsedAction]:
        """Parse an A2A QUERY message."""
        try:
            recipients = OllamaActionParser._parse_recipients(content)
            if not recipients:
                return None
            
            query_content = OllamaActionParser._parse_json_content(content, 'query_content')
            clinical_priority = re.search(r'clinical_priority:\s*(\d)', content)
            phi_present = re.search(r'phi_present:\s*(true|false)', content)
            reasoning = re.search(r'reasoning:\s*([^\n]+)', content)

            return ParsedAction(
                action_type=ActionType.A2A_QUERY,
                parameters={
                    "message_type": "QUERY",
                    "recipient_ids": recipients,
                    "query_content": query_content or {},
                    "clinical_priority": int(clinical_priority.group(1)) if clinical_priority else 3,
                    "phi_present": phi_present.group(1) == 'true' if phi_present else False
                },
                reasoning=reasoning.group(1).strip() if reasoning else None,
                confidence=0.8
            )
        except Exception:
            return None

    @staticmethod
    def _parse_a2a_response(content: str) -> Optional[ParsedAction]:
        """Parse an A2A RESPONSE message."""
        try:
            recipients = OllamaActionParser._parse_recipients(content)
            if not recipients:
                return None
            
            in_reply_to = re.search(r'in_reply_to:\s*([^\n]+)', content)
            response_content = OllamaActionParser._parse_json_content(content, 'response_content')
            clinical_priority = re.search(r'clinical_priority:\s*(\d)', content)
            phi_present = re.search(r'phi_present:\s*(true|false)', content)

            return ParsedAction(
                action_type=ActionType.A2A_RESPONSE,
                parameters={
                    "message_type": "RESPONSE",
                    "recipient_ids": recipients,
                    "in_reply_to": in_reply_to.group(1).strip() if in_reply_to else None,
                    "response_content": response_content or {},
                    "clinical_priority": int(clinical_priority.group(1)) if clinical_priority else 3,
                    "phi_present": phi_present.group(1) == 'true' if phi_present else False
                },
                confidence=0.9
            )
        except Exception:
            return None

    @staticmethod
    def _parse_a2a_notification(content: str) -> Optional[ParsedAction]:
        """Parse an A2A NOTIFICATION message."""
        try:
            recipients = OllamaActionParser._parse_recipients(content)
            if not recipients:
                return None
            
            notification_content = OllamaActionParser._parse_json_content(content, 'notification_content')
            priority = re.search(r'priority:\s*(LOW|NORMAL|HIGH|URGENT|CRITICAL)', content)
            requires_ack = re.search(r'requires_acknowledgment:\s*(true|false)', content)

            return ParsedAction(
                action_type=ActionType.A2A_NOTIFICATION,
                parameters={
                    "message_type": "NOTIFICATION",
                    "recipient_ids": recipients,
                    "notification_content": notification_content or {},
                    "priority": priority.group(1) if priority else "NORMAL",
                    "requires_acknowledgment": requires_ack.group(1) == 'true' if requires_ack else False
                },
                confidence=0.8
            )
        except Exception:
            return None

    @staticmethod
    def _parse_a2a_event(content: str) -> Optional[ParsedAction]:
        """Parse an A2A EVENT message."""
        try:
            recipients = OllamaActionParser._parse_recipients(content)
            if not recipients:
                return None
            
            event_content = OllamaActionParser._parse_json_content(content, 'event_content')
            clinical_priority = re.search(r'clinical_priority:\s*(\d)', content)
            requires_audit = re.search(r'requires_audit:\s*(true|false)', content)

            return ParsedAction(
                action_type=ActionType.A2A_EVENT,
                parameters={
                    "message_type": "EVENT",
                    "recipient_ids": recipients,
                    "event_content": event_content or {},
                    "clinical_priority": int(clinical_priority.group(1)) if clinical_priority else 3,
                    "requires_audit": requires_audit.group(1) == 'true' if requires_audit else True
                },
                confidence=0.8
            )
        except Exception:
            return None

    @staticmethod
    def _parse_json_action(
        json_data: Dict[str, Any],
        available_tools: Optional[Dict[str, Any]] = None,
        available_agents: Optional[List[str]] = None
    ) -> Optional[ParsedAction]:
        """Parse JSON formatted action from Ollama."""
        action_type = json_data.get("action_type")
        
        if action_type == "tool_call":
            return ParsedAction(
                action_type=ActionType.TOOL_CALL,
                target=json_data.get("tool_id"),
                parameters={
                    "tool_parameters": json_data.get("parameters", {}),
                    "send_result_to": json_data.get("send_result_to", [])
                },
                reasoning=json_data.get("reasoning"),
                confidence=json_data.get("confidence", 0.8)
            )
        elif action_type == "a2a_query":
            return ParsedAction(
                action_type=ActionType.A2A_QUERY,
                parameters={
                    "message_type": "QUERY",
                    "recipient_ids": json_data.get("recipient_ids", []),
                    "query_content": json_data.get("query_content", {}),
                    "clinical_priority": json_data.get("clinical_priority", 3),
                    "phi_present": json_data.get("phi_present", False)
                },
                reasoning=json_data.get("reasoning"),
                confidence=json_data.get("confidence", 0.8)
            )
        elif action_type == "a2a_response":
            return ParsedAction(
                action_type=ActionType.A2A_RESPONSE,
                parameters={
                    "message_type": "RESPONSE",
                    "recipient_ids": json_data.get("recipient_ids", []),
                    "in_reply_to": json_data.get("in_reply_to"),
                    "response_content": json_data.get("response_content", {}),
                    "clinical_priority": json_data.get("clinical_priority", 3),
                    "phi_present": json_data.get("phi_present", False)
                },
                reasoning=json_data.get("reasoning"),
                confidence=json_data.get("confidence", 0.9)
            )
        elif action_type == "a2a_notification":
            return ParsedAction(
                action_type=ActionType.A2A_NOTIFICATION,
                parameters={
                    "message_type": "NOTIFICATION",
                    "recipient_ids": json_data.get("recipient_ids", []),
                    "notification_content": json_data.get("notification_content", {}),
                    "priority": json_data.get("priority", "NORMAL"),
                    "requires_acknowledgment": json_data.get("requires_acknowledgment", False)
                },
                reasoning=json_data.get("reasoning"),
                confidence=json_data.get("confidence", 0.8)
            )
        elif action_type == "a2a_event":
            return ParsedAction(
                action_type=ActionType.A2A_EVENT,
                parameters={
                    "message_type": "EVENT",
                    "recipient_ids": json_data.get("recipient_ids", []),
                    "event_content": json_data.get("event_content", {}),
                    "clinical_priority": json_data.get("clinical_priority", 3),
                    "requires_audit": json_data.get("requires_audit", True)
                },
                reasoning=json_data.get("reasoning"),
                confidence=json_data.get("confidence", 0.8)
            )
        return None

    @staticmethod
    def convert_to_agent_action(
        parsed_action: ParsedAction,
        agent_context: Optional[Dict[str, Any]] = None
    ) -> Action:
        """
        Convert a parsed action to an executable agent Action.
        All actions result in A2A messages or tool execution.
        """
        if agent_context is None:
            agent_context = {}
        params = parsed_action.parameters or {}
        
        if parsed_action.action_type == ActionType.TOOL_CALL:
            return Action(
                action_type="execute_mcp_tool_and_send_results",
                parameters={
                    "tool_id": parsed_action.target,
                    "tool_parameters": params.get("tool_parameters", {}),
                    "send_result_to": params.get("send_result_to", []),
                    "reasoning": parsed_action.reasoning
                },
                priority=1,
                metadata={"confidence": parsed_action.confidence}
            )

        elif parsed_action.action_type == ActionType.A2A_QUERY:
            return Action(
                action_type="send_a2a_query",
                parameters=params,
                priority=2,
                metadata={
                    "confidence": parsed_action.confidence,
                    "reasoning": parsed_action.reasoning
                }
            )

        elif parsed_action.action_type == ActionType.A2A_RESPONSE:
            return Action(
                action_type="send_a2a_response",
                parameters=params,
                priority=1,  # High priority for responses
                metadata={
                    "confidence": parsed_action.confidence,
                    "reasoning": parsed_action.reasoning
                }
            )

        elif parsed_action.action_type == ActionType.A2A_NOTIFICATION:
            return Action(
                action_type="send_a2a_notification",
                parameters=params,
                priority=3,
                metadata={
                    "confidence": parsed_action.confidence,
                    "reasoning": parsed_action.reasoning
                }
            )

        elif parsed_action.action_type == ActionType.A2A_EVENT:
            return Action(
                action_type="send_a2a_event",
                parameters=params,
                priority=2,
                metadata={
                    "confidence": parsed_action.confidence,
                    "reasoning": parsed_action.reasoning
                }
            )

        else:
            # Fallback: create a notification for unknown actions
            return Action(
                action_type="send_a2a_notification",
                parameters={
                    "message_type": "NOTIFICATION",
                    "recipient_ids": agent_context.get("default_recipients", []),
                    "notification_content": {
                        "message": f"Unknown action: {parsed_action.action_type}",
                        "original_parameters": params
                    },
                    "priority": "LOW"
                },
                priority=5,
                metadata={"confidence": 0.3}
            )


class A2AMessageHelper:
    """Helper class for creating proper A2A messages from parsed actions."""
    
    @staticmethod
    def create_a2a_message_from_action(
        action: Action,
        sender_id: str,
        message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert an Action to a proper A2A message format.
        
        Args:
            action: The action to convert
            sender_id: ID of the sending agent
            message_id: Optional message ID (will generate if not provided)
            
        Returns:
            Dict representing A2A message ready to send
        """
        import uuid
        from datetime import datetime
        
        if message_id is None:
            message_id = str(uuid.uuid4())
        
        base_message = {
            "message_id": message_id,
            "sender_id": sender_id,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "confidence": action.metadata.get("confidence", 0.8),
                "reasoning": action.metadata.get("reasoning"),
                "action_type": action.action_type
            }
        }
        
        params = action.parameters
        
        if action.action_type == "send_a2a_query":
            base_message.update({
                "message_type": "QUERY",
                "recipient_ids": params.get("recipient_ids", []),
                "content": params.get("query_content", {}),
                "medical_metadata": {
                    "clinical_priority": params.get("clinical_priority", 3),
                    "phi_present": params.get("phi_present", False),
                    "requires_audit": True
                }
            })
            
        elif action.action_type == "send_a2a_response":
            base_message.update({
                "message_type": "RESPONSE",
                "recipient_ids": params.get("recipient_ids", []),
                "content": params.get("response_content", {}),
                "in_reply_to": params.get("in_reply_to"),
                "medical_metadata": {
                    "clinical_priority": params.get("clinical_priority", 3),
                    "phi_present": params.get("phi_present", False),
                    "requires_audit": True
                }
            })
            
        elif action.action_type == "send_a2a_notification":
            base_message.update({
                "message_type": "NOTIFICATION",
                "recipient_ids": params.get("recipient_ids", []),
                "content": params.get("notification_content", {}),
                "priority": params.get("priority", "NORMAL"),
                "medical_metadata": {
                    "requires_acknowledgment": params.get("requires_acknowledgment", False),
                    "requires_audit": True
                }
            })
            
        elif action.action_type == "send_a2a_event":
            base_message.update({
                "message_type": "EVENT",
                "recipient_ids": params.get("recipient_ids", []),
                "content": params.get("event_content", {}),
                "medical_metadata": {
                    "clinical_priority": params.get("clinical_priority", 3),
                    "requires_audit": params.get("requires_audit", True)
                }
            })
            
        return base_message
    
    @staticmethod
    def create_tool_result_message(
        tool_id: str,
        tool_result: Any,
        recipient_ids: List[str],
        sender_id: str,
        original_reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an A2A message containing tool execution results.
        
        Args:
            tool_id: ID of the executed tool
            tool_result: Result from tool execution
            recipient_ids: Who should receive the results
            sender_id: ID of the sending agent
            original_reasoning: Why the tool was called
            
        Returns:
            Dict representing A2A message with tool results
        """
        import uuid
        from datetime import datetime
        
        return {
            "message_id": str(uuid.uuid4()),
            "message_type": "NOTIFICATION",
            "sender_id": sender_id,
            "recipient_ids": recipient_ids,
            "content": {
                "tool_result": {
                    "tool_id": tool_id,
                    "result": tool_result,
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            "priority": "NORMAL",
            "created_at": datetime.utcnow().isoformat(),
            "medical_metadata": {
                "requires_audit": True,
                "tool_execution": True
            },
            "metadata": {
                "original_reasoning": original_reasoning,
                "result_type": "tool_execution"
            }
        }
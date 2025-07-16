

import asyncio
import json
import uvicorn
from fastapi import FastAPI
from typing import List, Dict, Any

from a2a.types import Message, TextPart, Role

# Import from the medical-a2a framework
from a2a_medical.base.agent import MedicalAgent, WorldModel, ProcessedObservation, Action, ActionResult, MentalState
from a2a_medical.integrations.ollama import OllamaReasoningMixin, OllamaActionParser
from a2a_medical.integrations.mcp import MCPDiscoveryMixin, MCPServer
from a2a.types import AgentCard, AgentCapabilities

# --- 1. Mock MCP Server ---
# This simulates an external medical service that provides tools.
mcp_app = FastAPI()

# A mock database of clinical trials
mock_clinical_trials = [
    {"trial_id": "NCT0428001", "condition": "Lung Cancer", "min_age": 50, "max_age": 75, "criteria": "non-small cell"},
    {"trial_id": "NCT0333123", "condition": "Diabetes", "min_age": 18, "max_age": 65, "criteria": "type 2"},
    {"trial_id": "NCT0555678", "condition": "Lung Cancer", "min_age": 40, "max_age": 80, "criteria": "smoker"}
]

@mcp_app.get("/tools")
async def get_tools():
    """MCP endpoint to advertise available tools."""
    return [{
        "name": "find_clinical_trials",
        "description": "Finds clinical trials based on condition, age, and other criteria.",
        "parameters": {
            "type": "object",
            "properties": {
                "condition": {"type": "string", "description": "The medical condition to search for."},
                "age": {"type": "integer", "description": "The patient's age."},
                "criteria": {"type": "string", "description": "Additional keywords for criteria matching."}
            },
            "required": ["condition", "age"]
        }
    }]

@mcp_app.post("/tools/find_clinical_trials/call")
async def call_find_clinical_trials(payload: Dict[str, Any]):
    """MCP endpoint to execute the tool call."""
    params = payload.get("parameters", {})
    condition = params.get("condition")
    age = params.get("age")
    criteria = params.get("criteria", "").lower()

    if not condition or age is None:
        return {"error": "Missing required parameters 'condition' or 'age'"}

    results = []
    for trial in mock_clinical_trials:
        if trial["condition"].lower() == condition.lower() and trial["min_age"] <= age <= trial["max_age"]:
            if not criteria or criteria in trial["criteria"].lower():
                results.append(trial)
    
    return {"results": results}

# --- 2. Upgraded Agent Implementation ---

class ClinicalTrialWorldModel(WorldModel):
    """A simple world model for our agent."""
    def __init__(self):
        super().__init__()
        self.patient_data = {}
        self.trial_results = []

    def update(self, observation: ProcessedObservation) -> None:
        if observation.observation_type == "patient_data":
            self.patient_data = observation.data
        elif observation.observation_type == "trial_results":
            self.trial_results = observation.data

    def query(self, query: str, context: Dict[str, Any] = None) -> Any:
        if query == "patient_data":
            return self.patient_data
        if query == "trial_results":
            return self.trial_results
        return None

    def predict(self, scenario: Dict[str, Any]) -> Any: return {}
    def get_state_summary(self) -> Dict[str, Any]: return {"patient_data": self.patient_data, "trial_results": self.trial_results}
    def reset(self) -> None:
        self.patient_data = {}
        self.trial_results = []

class ClinicalTrialAgent(OllamaReasoningMixin, MCPDiscoveryMixin, MedicalAgent):
    """An agent that uses reasoning to find clinical trials for a patient."""

    def __init__(self, agent_id: str, mcp_servers: List[MCPServer]):
        world_model = ClinicalTrialWorldModel()
        super().__init__(
            agent_id=agent_id,
            agent_type="clinical_trial_specialist",
            capabilities=["clinical_trial_matching"],
            world_model=world_model,
            mcp_servers=mcp_servers,
            model_name="llama3.1:8b", # Using a powerful local model
            ollama_temperature=0.0 # Low temperature for predictable, structured output
        )
        self.parser = OllamaActionParser()

    async def perceive(self, observation: Any) -> ProcessedObservation:
        """Perceive patient data from an incoming message."""
        if isinstance(observation, Message):
            # Assuming the message parts contain a JSON string with patient data
            try:
                patient_data = json.loads(observation.parts[0].text)
                return ProcessedObservation(data=patient_data, timestamp=asyncio.get_event_loop().time(), source="a2a_message", observation_type="patient_data")
            except (json.JSONDecodeError, IndexError):
                return ProcessedObservation(data={}, timestamp=asyncio.get_event_loop().time(), source="a2a_message", observation_type="error")
        return ProcessedObservation(data=observation, timestamp=asyncio.get_event_loop().time(), source="unknown")

    async def learn(self, state: MentalState, observation: ProcessedObservation) -> MentalState:
        """Update the world model with perceived data."""
        state.update_world_model(observation)
        return state

    async def reason(self, state: MentalState) -> Action:
        """The core multi-step reasoning logic."""
        patient_data = state.query_world_model("patient_data")
        trial_results = state.query_world_model("trial_results")

        if not patient_data:
            return Action(action_type="error", parameters={"message": "No patient data available to reason about."})

        # **Reasoning Step 1: If we don't have trial results, find them using a tool.**
        if not trial_results:
            system_prompt = f"""
You are a clinical trial specialist. Your task is to find a clinical trial for a patient.
First, analyze the patient data. Second, call the 'find_clinical_trials' tool with the correct parameters.
Patient Data: {json.dumps(patient_data)}
{self.parser.ACTION_FORMAT_PROMPT}
"""
            prompt = "Based on the patient data, what tool call should you make to find relevant clinical trials?"
        
        # **Reasoning Step 2: If we have results, interpret them and form a response.**
        else:
            system_prompt = f"""
You are a clinical trial specialist. You have received the following clinical trial options for the patient.
Your task is to analyze the results and formulate a final recommendation to send back to the requesting agent.
Patient Data: {json.dumps(patient_data)}
Trial Results: {json.dumps(trial_results)}
{self.parser.ACTION_FORMAT_PROMPT}
"""
            prompt = "Analyze the trial results and patient data. Formulate an A2A_RESPONSE message with your final recommendation. Explain why you chose a specific trial."

        # Use Ollama for reasoning
        ollama_output = await self.ollama_reason(prompt, system_prompt=system_prompt, include_tools=True)
        
        # Parse the LLM's output into a concrete action
        raw_response = ollama_output.get("response", "{}")
        try:
            # Ollama in JSON mode returns a string that needs to be parsed
            parsed_json = json.loads(raw_response)
            action_list = self.parser.parse_ollama_output(json.dumps(parsed_json.get("action")))
        except (json.JSONDecodeError, AttributeError):
            action_list = self.parser.parse_ollama_output(raw_response)

        if action_list:
            # Convert the first valid parsed action into an executable Agent Action
            return self.parser.convert_to_agent_action(action_list[0])
        
        return Action(action_type="error", parameters={"message": "Could not determine a valid action."})

    async def execute(self, action: Action) -> ActionResult:
        """Execute the action determined by the reasoning module."""
        if action.action_type == "execute_mcp_tool_and_send_results":
            tool_id = action.parameters.get("tool_id")
            tool_params = action.parameters.get("tool_parameters")
            try:
                print(f"🤖 Agent: Calling MCP tool '{tool_id}' with params: {tool_params}")
                result = await self.mcp_manager.call_tool(tool_id, tool_params)
                print(f"🛠️ MCP Server: Returned result: {result}")
                # This is a crucial step: update the world model with the tool's result
                # This triggers the next step in the reasoning cycle.
                self.mental_state.world_model.update(ProcessedObservation(data=result, timestamp=asyncio.get_event_loop().time(), source="mcp_tool", observation_type="trial_results"))
                return ActionResult(success=True, data={"status": "Tool executed, results stored in world model."})
            except Exception as e:
                return ActionResult(success=False, error=f"Tool call failed: {str(e)}")
        
        if action.action_type == "send_a2a_response":
            print(f"🤖 Agent: Formulating final A2A response.")
            # In a real system, this would send an A2A message. Here, we just format it.
            final_response = {
                "action": "A2A_RESPONSE",
                "recipient": action.parameters.get("recipient_ids"),
                "content": action.parameters.get("response_content")
            }
            return ActionResult(success=True, data=final_response)

        return ActionResult(success=False, error=f"Unknown action type: {action.action_type}")

    def build_agent_card(self) -> AgentCard:
        """Build the agent card for A2A discovery."""
        return AgentCard(
            name=self.agent_name,
            description=self.agent_description,
            version=self.agent_version,
            url=f"http://localhost:8001/{self.agent_id}", # Mock URL
            capabilities=AgentCapabilities(streaming=False),
            skills=[],
        )

# --- 3. Main Execution Logic ---

async def run_agent_logic(agent: ClinicalTrialAgent):
    """Simulates the full reasoning cycle of the agent."""
    # Patient data that would arrive in an A2A message
    patient_json = json.dumps({
        "patient_id": "P12345",
        "age": 55,
        "condition": "Lung Cancer",
        "notes": "Diagnosed with non-small cell lung cancer. Currently a smoker."
    })
    
    request_message = Message(messageId="msg-001", parts=[TextPart(text=patient_json)], role=Role.user)

    # --- First cycle: Find trials ---
    print("\n--- Agent Reasoning Cycle 1: Finding Trials ---")
    # The agent perceives the data, learns, reasons (decides to call a tool), and executes (calls the tool)
    await agent.process_request(request_message)
    
    # --- Second cycle: Interpret results and respond ---
    print("\n--- Agent Reasoning Cycle 2: Interpreting Results & Responding ---")
    # Now that the world model is updated with trial results, we trigger reasoning again.
    # The agent will now interpret the results and formulate a final response.
    final_action = await agent.reason(agent.mental_state)
    final_result = await agent.execute(final_action)

    print("\n--- Final Agent Output ---")
    print(json.dumps(final_result.data, indent=2))


async def main():
    """Sets up the mock server and runs the agent."""
    # Setup for the mock MCP server
    config = uvicorn.Config(mcp_app, host="127.0.0.1", port=8082, log_level="info")
    server = uvicorn.Server(config)
    
    print("🚀 Starting Mock MCP Server...")
    mcp_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1) # Give server time to start

    # Setup the agent
    mcp_server_config = [MCPServer(name="trial_finder", url="http://127.0.0.1:8082", description="Finds clinical trials.")]
    agent = ClinicalTrialAgent("trial-agent-001", mcp_server_config)
    await agent.mcp_manager.register_server(mcp_server_config[0]) # Manually register server

    try:
        await run_agent_logic(agent)
    finally:
        print("\n🛑 Stopping Mock MCP Server...")
        await server.shutdown()
        mcp_task.cancel()

if __name__ == "__main__":
    # Note: This example requires fastapi and uvicorn.
    # You can install them with: pip install "fastapi[all]"
    asyncio.run(main())

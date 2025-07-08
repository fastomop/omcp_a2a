"""
Example: Building an LLM-Powered Medical Diagnostic Agent
using the A2A Medical Foundation Framework

This example shows how to create a medical AI agent that:
1. Uses an LLM (like OpenAI GPT, Anthropic Claude, etc.) for medical reasoning
2. Maintains medical safety and compliance
3. Can communicate with other medical agents
4. Follows proper medical protocols
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import openai  # You can replace this with any LLM API

# Import the A2A Medical Framework
from a2a_medical.base.agent import MedicalAgent, WorldModel, ProcessedObservation, Action, ActionResult
from a2a_medical.models.medical import Patient, MedicalRecord
from a2a_medical.validators.safety import SafetyValidator
from a2a_medical.exceptions import ValidationError
from a2a.types import AgentCard, AgentCapabilities


class MedicalWorldModel(WorldModel):
    """
    A World Model that stores medical knowledge and patient information.
    This is like the agent's "memory" and "knowledge base".
    """
    
    def __init__(self):
        super().__init__()
        self.patient_database: Dict[str, Patient] = {}
        self.medical_records: Dict[str, List[MedicalRecord]] = {}
        self.knowledge_base: Dict[str, Any] = {}
        self.last_updated = None
    
    def update(self, observation: ProcessedObservation) -> None:
        """Update the world model with new medical information."""
        if observation.source == "patient_data":
            # Store patient information
            patient_id = observation.data.get("patient_id")
            if patient_id:
                self.knowledge_base[f"patient_{patient_id}"] = observation.data
        
        elif observation.source == "medical_record":
            # Store medical records
            record = observation.data
            patient_id = record.get("patient_id")
            if patient_id:
                if patient_id not in self.medical_records:
                    self.medical_records[patient_id] = []
                self.medical_records[patient_id].append(record)
        
        self.last_updated = observation.timestamp
    
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Query medical knowledge from the world model."""
        # Simple keyword-based search (you could make this much more sophisticated)
        results = {}
        
        for key, data in self.knowledge_base.items():
            if any(word.lower() in str(data).lower() for word in query.split()):
                results[key] = data
        
        return results if results else "No relevant medical information found"
    
    def predict(self, scenario: Dict[str, Any]) -> Any:
        """Make medical predictions based on current knowledge."""
        patient_id = scenario.get("patient_id")
        
        if patient_id and patient_id in self.medical_records:
            records = self.medical_records[patient_id]
            return {
                "prediction": "Based on medical history",
                "confidence": 0.75,
                "records_analyzed": len(records)
            }
        
        return {"prediction": "Insufficient data", "confidence": 0.0}
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current world model state."""
        return {
            "patients_tracked": len(self.patient_database),
            "total_records": sum(len(records) for records in self.medical_records.values()),
            "knowledge_items": len(self.knowledge_base),
            "last_updated": self.last_updated
        }
    
    def reset(self) -> None:
        """Reset the world model."""
        self.patient_database.clear()
        self.medical_records.clear()
        self.knowledge_base.clear()
        self.last_updated = None


class LLMPoweredDiagnosticAgent(MedicalAgent):
    """
    A Medical Diagnostic Agent powered by an LLM (Large Language Model).
    
    This agent:
    - Uses an LLM for medical reasoning and diagnosis
    - Maintains medical safety and compliance
    - Can communicate with other medical agents
    - Follows the PCE (Perception-Cognition-Execution) architecture
    """
    
    def __init__(self, agent_id: str, llm_api_key: str, llm_model: str = "gpt-4"):
        # Create our medical world model
        world_model = MedicalWorldModel()
        
        # Initialize the medical agent
        super().__init__(
            agent_id=agent_id,
            agent_type="llm_diagnostic",
            capabilities=[
                "symptom_analysis", 
                "differential_diagnosis", 
                "treatment_recommendations",
                "medical_consultation"
            ],
            world_model=world_model,
            agent_name=f"LLM Diagnostic Agent {agent_id}",
            agent_description="AI-powered diagnostic agent using large language models"
        )
        
        # Setup LLM integration
        self.llm_client = openai.OpenAI(api_key=llm_api_key)
        self.llm_model = llm_model
        
        # Initialize safety validator
        self.safety_validator = self._create_safety_validator()
        
        # Medical knowledge prompts
        self.system_prompt = """You are a medical diagnostic AI assistant. You must:
        1. Provide accurate medical information based on evidence
        2. Always recommend consulting with healthcare professionals
        3. Never provide emergency medical advice - direct to emergency services
        4. Follow medical ethics and patient confidentiality
        5. Admit uncertainty when appropriate
        
        You have access to patient information and medical records through the world model.
        Provide differential diagnoses with confidence levels and recommend appropriate tests.
        """
    
    def _create_safety_validator(self):
        """Create a concrete safety validator for medical validation."""
        class ConcreteSafetyValidator(SafetyValidator):
            def __init__(self):
                super().__init__()
                self.dangerous_keywords = [
                    "self-medicate", "ignore symptoms", "definitely cancer",
                    "certainly fatal", "avoid all doctors"
                ]
            
            async def validate_query_safety(self, query, context=None):
                # Check for dangerous content
                dangerous = any(keyword in query.lower() for keyword in self.dangerous_keywords)
                return {
                    "is_safe": not dangerous,
                    "risk_level": "high" if dangerous else "low",
                    "warnings": ["Potentially unsafe content detected"] if dangerous else []
                }
            
            async def validate_response_safety(self, response, context=None):
                # Check response safety
                dangerous = any(keyword in response.lower() for keyword in self.dangerous_keywords)
                return {
                    "is_safe": not dangerous,
                    "risk_level": "high" if dangerous else "low",
                    "recommendations": ["Review response before sending"] if dangerous else []
                }
        
        return ConcreteSafetyValidator()
    
    async def perceive(self, observation: Any) -> ProcessedObservation:
        """
        Perception: Process incoming medical data and messages.
        This is where the agent "sees" and understands medical information.
        """
        
        # Handle different types of medical observations
        if isinstance(observation, dict):
            # Medical data input
            data = observation
            source = data.get("source", "medical_input")
            confidence = 0.9
            
        elif hasattr(observation, 'parts'):
            # A2A message input
            # Extract text from message parts
            text_content = ""
            for part in observation.parts:
                if hasattr(part, 'text'):
                    text_content += part.text + " "
            
            data = {
                "message_content": text_content.strip(),
                "message_id": getattr(observation, 'messageId', None),
                "timestamp": datetime.now().isoformat()
            }
            source = "a2a_message"
            confidence = 0.85
            
        else:
            # Raw text or other input
            data = {"content": str(observation)}
            source = "raw_input"
            confidence = 0.7
        
        # Validate the input for safety
        safety_result = await self.safety_validator.validate_query_safety(str(data))
        if not safety_result["is_safe"]:
            raise ValidationError("Unsafe medical query detected", field="observation")
        
        return ProcessedObservation(
            data=data,
            timestamp=datetime.now().timestamp(),
            source=source,
            confidence=confidence
        )
    
    async def learn(self, state, observation: ProcessedObservation):
        """
        Learning: Update the world model with new medical information.
        This is where the agent updates its knowledge and memory.
        """
        
        # Update the world model with the new observation
        state.update_world_model(observation)
        
        # Extract any patient information
        if "patient" in str(observation.data).lower():
            # This could trigger patient record updates
            state.add_goal("update_patient_records")
        
        # Add to memory for context
        state.memory[f"observation_{observation.timestamp}"] = observation.data
        
        # Set emotional state based on observation urgency
        if any(urgent_word in str(observation.data).lower() 
               for urgent_word in ["emergency", "urgent", "critical", "severe"]):
            state.set_emotion("urgency", 0.8)
        else:
            state.set_emotion("calm", 0.6)
        
        return state
    
    async def reason(self, state) -> Action:
        """
        Reasoning: Use LLM to analyze medical information and make decisions.
        This is where the medical AI "thinks" about the problem.
        """
        
        # Gather context from world model
        recent_observations = list(state.memory.values())[-5:]  # Last 5 observations
        world_summary = state.world_model.get_state_summary()
        
        # Create medical reasoning prompt
        context = {
            "recent_observations": recent_observations,
            "world_state": world_summary,
            "goals": state.goals,
            "emotions": state.emotions
        }
        
        medical_prompt = f"""
        Based on the following medical context, provide a diagnostic assessment:
        
        Recent Medical Information:
        {json.dumps(recent_observations, indent=2)}
        
        World Model State:
        {json.dumps(world_summary, indent=2)}
        
        Current Goals: {state.goals}
        
        Please provide:
        1. Differential diagnosis with confidence levels
        2. Recommended diagnostic tests or procedures
        3. Immediate actions needed
        4. When to seek emergency care
        
        Response format: JSON with diagnostic_assessment, recommendations, urgency_level
        """
        
        try:
            # Call LLM for medical reasoning
            response = await self._call_llm(medical_prompt)
            
            # Parse LLM response
            reasoning_result = self._parse_llm_response(response)
            
            # Determine action type based on reasoning
            if reasoning_result.get("urgency_level", "low") == "emergency":
                action_type = "emergency_referral"
                priority = 1
            elif reasoning_result.get("diagnostic_assessment"):
                action_type = "provide_diagnosis"
                priority = 2
            else:
                action_type = "request_more_information"
                priority = 3
            
            return Action(
                action_type=action_type,
                parameters={
                    "llm_reasoning": reasoning_result,
                    "confidence": reasoning_result.get("confidence", 0.5)
                },
                priority=priority
            )
            
        except Exception as e:
            # Fallback action if LLM fails
            return Action(
                action_type="system_error",
                parameters={"error": str(e)},
                priority=5
            )
    
    async def execute(self, action: Action) -> ActionResult:
        """
        Execution: Carry out medical actions and provide responses.
        This is where the agent "acts" on its medical decisions.
        """
        
        try:
            if action.action_type == "provide_diagnosis":
                # Generate medical response
                llm_reasoning = action.parameters.get("llm_reasoning", {})
                
                response_data = {
                    "diagnosis": llm_reasoning.get("diagnostic_assessment", "Assessment pending"),
                    "recommendations": llm_reasoning.get("recommendations", []),
                    "confidence": action.parameters.get("confidence", 0.5),
                    "next_steps": llm_reasoning.get("next_steps", []),
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.agent_id
                }
                
                # Validate response safety
                safety_check = await self.safety_validator.validate_response_safety(
                    str(response_data)
                )
                
                if not safety_check["is_safe"]:
                    response_data["safety_warning"] = "Response flagged for review"
                
                success = True
                error = None
                
            elif action.action_type == "emergency_referral":
                response_data = {
                    "urgent_notice": "EMERGENCY CONDITION DETECTED",
                    "action": "Immediate medical attention required",
                    "instructions": "Contact emergency services or go to nearest emergency room",
                    "agent_note": "This is an automated assessment - seek immediate professional care"
                }
                success = True
                error = None
                
            elif action.action_type == "request_more_information":
                response_data = {
                    "request": "Additional information needed for accurate assessment",
                    "questions": [
                        "Can you provide more details about symptoms?",
                        "What is the timeline of symptom development?",
                        "Any relevant medical history?"
                    ]
                }
                success = True
                error = None
                
            else:
                response_data = {"status": "Action completed", "action": action.action_type}
                success = True
                error = None
            
            return ActionResult(
                success=success,
                data=response_data,
                error=error,
                metadata={
                    "action_type": action.action_type,
                    "execution_time": datetime.now().isoformat(),
                    "agent_version": self.agent_version
                }
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                data=None,
                error=f"Execution failed: {str(e)}",
                metadata={"action_type": action.action_type}
            )
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API for medical reasoning."""
        try:
            response = await self.llm_client.chat.completions.acreate(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for medical accuracy
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured medical data."""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                return json.loads(response)
            else:
                # Fallback: structure the text response
                return {
                    "diagnostic_assessment": response,
                    "confidence": 0.6,
                    "urgency_level": "low"
                }
        except json.JSONDecodeError:
            return {
                "diagnostic_assessment": response,
                "confidence": 0.5,
                "urgency_level": "low"
            }
    
    def build_agent_card(self) -> AgentCard:
        """Build the A2A agent card for this medical agent."""
        return AgentCard(
            name=self.agent_name,
            description=self.agent_description,
            url=f"https://medical-agents.local/llm-diagnostic/{self.agent_id}",
            version=self.agent_version,
            capabilities=AgentCapabilities(streaming=True),
            skills=self.capabilities,
            defaultInputModes=["text", "structured_medical_data"],
            defaultOutputModes=["text", "json", "medical_report"]
        )


# Example Usage
async def demo_llm_medical_agent():
    """Demonstrate how to use the LLM-powered medical agent."""
    
    print("🏥 A2A Medical Foundation Framework - LLM Agent Demo")
    print("=" * 60)
    
    # Create the agent
    agent = LLMPoweredDiagnosticAgent(
        agent_id="llm-diag-001",
        llm_api_key="your-llm-api-key-here",  # Replace with actual API key
        llm_model="gpt-4"
    )
    
    print(f"✅ Created agent: {agent.agent_name}")
    print(f"🔧 Capabilities: {', '.join(agent.capabilities)}")
    
    # Example 1: Process a medical query
    print("\n📝 Example 1: Processing a medical query...")
    
    medical_query = {
        "patient_symptoms": ["fever", "cough", "fatigue"],
        "duration": "3 days",
        "patient_age": 35,
        "source": "patient_data"
    }
    
    try:
        # Process through the PCE cycle
        observation = await agent.perceive(medical_query)
        print(f"👁️  Perception: {observation.source} (confidence: {observation.confidence})")
        
        updated_state = await agent.learn(agent.mental_state, observation)
        print(f"🧠 Learning: Updated world model with {len(updated_state.memory)} items")
        
        action = await agent.reason(updated_state)
        print(f"💭 Reasoning: Decided on '{action.action_type}' (priority: {action.priority})")
        
        result = await agent.execute(action)
        print(f"⚡ Execution: {'✅ Success' if result.success else '❌ Failed'}")
        
        if result.success:
            print("\n📋 Medical Assessment:")
            diagnosis = result.data.get("diagnosis", "No diagnosis provided")
            print(f"   Diagnosis: {diagnosis}")
            recommendations = result.data.get("recommendations", [])
            if recommendations:
                print("   Recommendations:")
                for rec in recommendations:
                    print(f"   - {rec}")
    
    except Exception as e:
        print(f"❌ Error processing medical query: {str(e)}")
    
    # Example 2: Agent information
    print(f"\n🤖 Agent Information:")
    agent_info = agent.get_agent_info()
    print(f"   ID: {agent_info['agent_id']}")
    print(f"   Type: {agent_info['agent_type']}")
    print(f"   Status: {'🟢 Active' if agent_info['is_active'] else '🔴 Inactive'}")
    
    print("\n✨ Demo completed!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_llm_medical_agent()) 
# A2A Medical Foundation Framework

A comprehensive **abstract foundation framework** for building secure, compliant, and efficient medical Agent-to-Agent (A2A) systems using PCE-based agentic architectures.

## 🏗️ Architecture Philosophy

This framework provides **abstract base classes** rather than concrete implementations, allowing to build specialized medical A2A systems tailored to your specific needs while ensuring compliance and safety standards.

### Core Architecture: PCE (Perception-Cognition-Execution)

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Perception  │───▶│  Cognition   │───▶│ Execution   │
│ (P)         │    │  (C)         │    │ (E)         │
│             │    │              │    │             │
│ Process     │    │ World Model  │    │ Execute     │
│ Observations│    │ + Learning   │    │ Actions     │
│             │    │ + Reasoning  │    │             │
└─────────────┘    └──────────────┘    └─────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd medical-a2a

# Install dependencies using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev,test]"

# Alternative: Using pip
pip install -e .
```

### Creating Your First Medical Agent

```python
import asyncio
from a2a_medical import MedicalAgent, WorldModel
from a2a_medical.base.agent import ProcessedObservation, Action, ActionResult
from a2a.types import AgentCard, Message

# 1. Implement your World Model
class MyMedicalWorldModel(WorldModel):
    def __init__(self):
        super().__init__()
        self.knowledge_base = {}
        self.patient_history = {}
    
    def update(self, observation: ProcessedObservation) -> None:
        # Update your world model with new medical observations
        self.knowledge_base[observation.source] = observation.data
        self.last_updated = observation.timestamp
    
    def query(self, query: str, context: dict = None) -> any:
        # Query medical knowledge
        return self.knowledge_base.get(query, "No information available")
    
    def predict(self, scenario: dict) -> any:
        # Make medical predictions
        return {"prediction": "example_outcome", "confidence": 0.85}
    
    def get_state_summary(self) -> dict:
        return {
            "knowledge_items": len(self.knowledge_base),
            "patients_tracked": len(self.patient_history)
        }
    
    def reset(self) -> None:
        self.knowledge_base.clear()
        self.patient_history.clear()

# 2. Implement your Medical Agent
class DiagnosticAgent(MedicalAgent):
    def __init__(self, agent_id: str):
        world_model = MyMedicalWorldModel()
        super().__init__(
            agent_id=agent_id,
            agent_type="diagnostic",
            capabilities=["symptom_analysis", "differential_diagnosis"],
            world_model=world_model
        )
    
    async def perceive(self, observation: any) -> ProcessedObservation:
        # Process incoming medical data
        return ProcessedObservation(
            data=observation,
            timestamp=datetime.now(),
            source="medical_input",
            confidence=0.9
        )
    
    async def learn(self, state, observation: ProcessedObservation):
        # Update world model and mental state
        state.update_world_model(observation)
        return state
    
    async def reason(self, state) -> Action:
        # Medical reasoning and decision making
        return Action(
            action_type="provide_diagnosis",
            parameters={"diagnosis": "example_condition"},
            priority=1
        )
    
    async def execute(self, action: Action) -> ActionResult:
        # Execute medical actions
        return ActionResult(
            success=True,
            data={"response": f"Executed {action.action_type}"},
            metadata=action.parameters
        )
    
    def build_agent_card(self) -> AgentCard:
        # Define your agent's A2A capabilities
        return AgentCard(
            name=f"diagnostic-agent-{self.agent_id}",
            description="Medical diagnostic agent with symptom analysis",
            url=f"https://medical-agents.local/diagnostic/{self.agent_id}",
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=True),
            skills=[],
            defaultInputModes=["text"],
            defaultOutputModes=["text"]
        )

# 3. Use your agent
async def main():
    agent = DiagnosticAgent("diag-001")
    
    # Process a medical request
    sample_message = Message(
        messageId="msg-123",
        parts=[],
        role=Role.user
    )
    
    response = await agent.process_request(sample_message)
    print("Agent response received!")

# Run the example
# asyncio.run(main())
```

## 🏥 Framework Components

### Abstract Base Classes

The framework provides abstract foundations in several categories:

#### 🤖 **Core Agent Architecture**
- `MedicalAgent` - Base medical agent with PCE architecture
- `WorldModel` - Abstract world modeling for medical knowledge
- `MentalState` - Agent mental state management
- `CognitiveModule` - Pluggable cognitive capabilities

#### 🔗 **Protocols & Communication**
- `MedicalA2AProtocol` - Medical A2A protocol extensions
- `ComplianceProtocol` - Healthcare compliance standards
- `EmergencyProtocol` - Emergency response procedures
- `AgentDiscovery` - Agent discovery and registration
- `MessageRouter` - Intelligent message routing

#### ✅ **Validation & Safety**
- `SafetyValidator` - Medical safety validation
- `DrugInteractionValidator` - Medication safety checking
- `DosageValidator` - Dosage validation systems
- `EmergencyValidator` - Emergency condition detection
- `ComplianceValidator` - Regulatory compliance

#### 📊 **Monitoring & Logging**
- `MedicalLogger` - PHI-protected medical logging
- `MetricsCollector` - Performance and compliance metrics
- `SecurityMetricsCollector` - Security event monitoring
- `ComplianceMonitor` - Compliance tracking

## 🛡️ Compliance & Security

### Supported Standards
- **HIPAA** - Health Insurance Portability and Accountability Act
- **GDPR** - General Data Protection Regulation
- **HITECH** - Health Information Technology for Economic and Clinical Health
- **FDA 21 CFR Part 11** - Electronic Records and Signatures

### Security Features
- PHI (Protected Health Information) detection and protection
- Encryption abstractions for data at rest and in transit
- Audit logging for compliance tracking
- Access control frameworks
- Secure communication protocols

## 📋 Example Use Cases

### 1. Telemedicine Platform
```python
class TelemedicineAgent(MedicalAgent):
    # Implement virtual consultation capabilities
    # Handle patient-provider communication
    # Ensure HIPAA compliance
```

### 2. Clinical Decision Support
```python
class ClinicalDSSAgent(MedicalAgent):
    # Implement evidence-based recommendations
    # Integration with clinical guidelines
    # Drug interaction checking
```

### 3. Medical Record Integration
```python
class EHRIntegrationAgent(MedicalAgent):
    # Implement HL7 FHIR compatibility
    # Cross-system data synchronization
    # Audit trail maintenance
```

### 4. Emergency Response System
```python
class EmergencyAgent(MedicalAgent):
    # Implement critical condition detection
    # Automated emergency protocols
    # Priority-based routing
```

## 🔧 Development

### Project Structure
```
a2a-medical-foundation/
├── src/a2a_medical/
│   ├── base/              # Core abstract agents
│   ├── models/            # Data models (Pydantic)
│   ├── protocols/         # A2A protocols
│   ├── validators/        # Safety & compliance
│   ├── utils/            # Logging, metrics, crypto
│   └── exceptions.py     # Custom exceptions
├── tests/                # Test suites
├── docs/                 # Documentation
└── examples/             # Usage examples
```

### Key Design Principles

1. **Abstract Foundation** - Provides structure without imposing implementation
2. **Medical Domain Focus** - Built specifically for healthcare applications
3. **Compliance First** - Designed with regulatory requirements in mind
4. **Safety Critical** - Multiple validation layers for patient safety
5. **Extensible Architecture** - Easy to customize and extend
6. **Agent-Based** - Supports multi-agent medical ecosystems

## 📚 Documentation

- [Architecture Guide](docs/architecture.md) - Deep dive into the framework design
- [Compliance Guide](docs/compliance.md) - Meeting healthcare regulations
- [Security Guide](docs/security.md) - Implementing secure medical systems
- [API Reference](docs/api.md) - Complete API documentation
- [Examples](examples/) - Working examples and tutorials

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚖️ Medical Disclaimer

This framework is designed for building medical software systems but does not provide medical advice. Any implementations must be validated by qualified medical professionals and comply with applicable regulations before use in clinical environments.

---

**Built for the future of medical interoperability** 🏥✨

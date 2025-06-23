# A2A Medical Foundation Framework - Test Suite

This directory contains comprehensive tests for the A2A Medical Foundation Framework. The test suite is designed to ensure the reliability, safety, and compliance of all framework components.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                # Shared fixtures and configuration
├── test_base_agent.py         # Base agent component tests
├── test_validators.py         # Validator component tests
├── test_utils.py              # Utility component tests
├── test_protocols.py          # Protocol component tests
├── test_models.py             # Model component tests
├── test_integration.py        # Integration tests
└── README.md                  # This file
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- **Purpose**: Test individual components in isolation
- **Coverage**: Abstract class validation, concrete implementations, data models
- **Focus**: Individual method behaviors, edge cases, error handling

### Integration Tests (`@pytest.mark.integration`)
- **Purpose**: Test component interactions and workflows
- **Coverage**: Agent-to-agent communication, world model updates, protocol flows
- **Focus**: End-to-end scenarios, multi-component workflows

### Safety Tests (`@pytest.mark.safety`)
- **Purpose**: Validate safety-critical functionality
- **Coverage**: Safety validators, emergency protocols, drug interactions
- **Focus**: Medical safety, patient protection, error prevention

### Compliance Tests (`@pytest.mark.compliance`)
- **Purpose**: Ensure regulatory compliance
- **Coverage**: HIPAA, GDPR, FDA validation, audit trails
- **Focus**: Data protection, audit logging, regulatory requirements

### Slow Tests (`@pytest.mark.slow`)
- **Purpose**: Long-running tests that stress the system
- **Coverage**: Large data processing, extended workflows
- **Focus**: Performance, scalability, resource usage

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --type unit
python run_tests.py --type integration
python run_tests.py --type safety
python run_tests.py --type compliance

# Run fast tests only (exclude slow tests)
python run_tests.py --type fast
```

### Advanced Options

```bash
# Run with coverage
python run_tests.py --coverage

# Generate HTML coverage report
python run_tests.py --html-cov

# Run tests in parallel
python run_tests.py --parallel

# Verbose output
python run_tests.py --verbose

# Stop on first failure
python run_tests.py --failfast
```

### Direct pytest Usage

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_base_agent.py

# Run specific test class
pytest tests/test_base_agent.py::TestWorldModelAbstract

# Run specific test method
pytest tests/test_base_agent.py::TestWorldModelAbstract::test_cannot_instantiate_abstract_world_model

# Run with markers
pytest -m "unit and not slow"
pytest -m "safety or compliance"

# Run with coverage
pytest --cov=src/a2a_medical --cov-report=html
```

## Test Components

### Base Agent Tests (`test_base_agent.py`)

Tests for the foundational agent architecture:

- **WorldModel**: Abstract base class and concrete implementations
- **MedicalAgent**: Agent lifecycle (perceive, learn, reason, execute)
- **MentalState**: Agent state management
- **CognitiveModule**: Reasoning and decision-making components

Key test scenarios:
- Abstract class instantiation prevention
- Perception-Cognition-Execution (PCE) cycle
- World model state updates and queries
- Learning from observation-action-outcome sequences

### Validator Tests (`test_validators.py`)

Tests for safety and compliance validation:

- **SafetyValidator**: Medical query and response safety
- **ComplianceValidator**: HIPAA, GDPR, and other regulatory compliance
- **DrugInteractionValidator**: Medication safety and contraindications
- **EmergencyValidator**: Emergency situation detection and response

Key test scenarios:
- Safety rule configuration and enforcement
- Compliance checking across different standards
- Drug interaction detection
- Emergency level assessment

### Utility Tests (`test_utils.py`)

Tests for framework utilities:

- **MedicalLogger**: PHI-protected logging and audit trails
- **MetricsCollector**: Performance and compliance metrics
- **CryptoManager**: Encryption and security utilities

Key test scenarios:
- PHI detection and sanitization
- Audit trail generation and search
- Metrics collection and aggregation
- Encryption/decryption operations

### Protocol Tests (`test_protocols.py`)

Tests for A2A communication protocols:

- **MedicalA2AProtocol**: Medical-specific communication patterns
- **AgentDiscovery**: Medical agent discovery and registration
- **MessageRouter**: Message routing and delivery
- **ComplianceProtocol**: Compliance-aware messaging

Key test scenarios:
- Agent discovery by medical specialty
- Medical capability matching
- Emergency escalation protocols
- Compliance validation in messaging

### Model Tests (`test_models.py`)

Tests for data models and structures:

- **Patient**: Patient information and validation
- **Provider**: Healthcare provider data
- **MedicalRecord**: Medical record creation and validation
- **A2AMessage**: Inter-agent messaging
- **MedicalTask**: Task management and workflow

Key test scenarios:
- Data validation and serialization
- Model field constraints and requirements
- JSON round-trip integrity
- Business rule enforcement

### Integration Tests (`test_integration.py`)

Tests for complete workflows and scenarios:

- **Multi-agent collaboration**: Triage and specialist consultation
- **Emergency workflows**: Critical patient scenarios
- **Compliance integration**: End-to-end compliance checking
- **World model predictions**: Learning and prediction capabilities

Key test scenarios:
- Complete patient assessment workflows
- A2A message-driven consultations
- Multi-agent emergency response
- Compliance and safety integration

## Test Data and Fixtures

### Shared Fixtures (`conftest.py`)

The `conftest.py` file provides shared fixtures used across tests:

- **Medical Agents**: Concrete implementations for testing
- **Sample Data**: Patients, providers, medical records
- **Mock Objects**: Simulated external dependencies
- **Test Configuration**: Common test settings

### Test Markers

Tests are categorized using pytest markers:

```python
@pytest.mark.unit
def test_individual_component():
    pass

@pytest.mark.integration
def test_component_interaction():
    pass

@pytest.mark.safety
def test_safety_critical_feature():
    pass

@pytest.mark.compliance
def test_regulatory_compliance():
    pass

@pytest.mark.slow
def test_performance_scenario():
    pass

@pytest.mark.asyncio
async def test_async_functionality():
    pass
```

## Coverage Goals

### Coverage Targets
- **Overall Coverage**: >90%
- **Critical Components**: >95%
- **Safety Components**: 100%
- **Abstract Classes**: 100% (instantiation prevention)

### Coverage Reports

```bash
# Generate terminal coverage report
pytest --cov=src/a2a_medical --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=src/a2a_medical --cov-report=html

# Generate XML coverage report (for CI)
pytest --cov=src/a2a_medical --cov-report=xml
```

View HTML coverage report:
```bash
python -m http.server 8000 --directory htmlcov
```
Then open http://localhost:8000 in your browser.

## Continuous Integration

### GitHub Actions

The test suite runs automatically on:
- Pull requests to `main` and `develop` branches
- Pushes to `main` and `develop` branches

CI pipeline includes:
1. **Test Matrix**: Multiple Python versions and test types
2. **Linting**: Black, flake8, mypy
3. **Security**: Bandit security linting, safety checks
4. **Documentation**: Sphinx documentation build
5. **Coverage**: Codecov integration

### Local Pre-commit Hooks

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

This ensures code quality checks run before each commit.

## Writing New Tests

### Test Guidelines

1. **Follow naming conventions**: `test_*.py` files, `test_*` functions
2. **Use descriptive names**: Test names should explain what is being tested
3. **Document test purpose**: Include docstrings explaining test scenarios
4. **Use appropriate markers**: Mark tests with correct categories
5. **Mock external dependencies**: Use fixtures and mocks for isolation
6. **Test edge cases**: Include boundary conditions and error scenarios

### Test Template

```python
import pytest
from unittest.mock import Mock, AsyncMock

from a2a_medical.your_module import YourClass


class TestYourClass:
    """Test the YourClass component."""
    
    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic functionality works as expected."""
        # Arrange
        instance = YourClass("test-id")
        
        # Act
        result = instance.do_something()
        
        # Assert
        assert result is not None
        assert result.status == "success"
    
    @pytest.mark.unit
    def test_error_handling(self):
        """Test proper error handling for invalid input."""
        instance = YourClass("test-id")
        
        with pytest.raises(ValueError, match="Invalid input"):
            instance.do_something_invalid()
    
    @pytest.mark.integration
    def test_integration_scenario(self, sample_data_fixture):
        """Test integration with other components."""
        # Integration test implementation
        pass
```

### Best Practices

1. **Arrange-Act-Assert**: Structure tests clearly
2. **Single Responsibility**: Each test should test one thing
3. **Deterministic**: Tests should produce consistent results
4. **Independent**: Tests should not depend on each other
5. **Fast**: Unit tests should run quickly
6. **Readable**: Tests should be easy to understand and maintain

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the package is installed in development mode
   ```bash
   uv pip install -e .
   ```

2. **Missing Dependencies**: Install test dependencies
   ```bash
   uv pip install -e ".[test]"
   ```

3. **Abstract Class Errors**: These are expected for abstract base classes

4. **Async Test Issues**: Use `@pytest.mark.asyncio` for async tests

### Getting Help

- Check the test output for specific error messages
- Review the test documentation and examples
- Run tests with `--verbose` for detailed output
- Use `--pdb` to drop into debugger on failures

## Contributing

When contributing new tests:

1. Follow the existing test structure and patterns
2. Add appropriate markers for test categorization
3. Include both positive and negative test cases
4. Update this README if adding new test categories
5. Ensure all tests pass before submitting pull requests

The test suite is a critical component for ensuring the safety and reliability of medical AI systems. Please take care to write thorough, well-documented tests. 
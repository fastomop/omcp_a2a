# OMCP: OMOP Model Context Protocol

OMCP is a Model Context Protocol (MCP) server implementation for the Observational Medical Outcomes Partnership (OMOP) Common Data Model. It enables LLMs to query and analyze healthcare data stored in OMOP format through standardized interfaces.

## 🌟 Features

- **Natural Language to SQL**: Convert healthcare questions into OMOP-compliant SQL
- **MCP Architecture**: Modular MCP servers for database operations, validation, and LLM interactions
- **A2A Protocol Support**: Agent-to-Agent communication for external tool integration
- **SQL Validation & Refinement**: Automatic validation and correction of SQL against OMOP CDM rules
- **Multiple Database Support**: Works with PostgreSQL, DuckDB, and other OMOP-compatible databases

## 🏗️ Architecture

OMCP employs a modular architecture with specialized MCP servers:

```
                 ┌─────────────┐
                 │   Client    │
                 │ Applications│
                 └─────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ Orchestrator │
                └───┬───┬───┬──┘
                    │   │   │
        ┌───────────┘   │   └────────────┐
        │               │                │
        ▼               ▼                ▼
┌───────────────┐ ┌──────────────┐ ┌──────────────┐
│  SQL Server   │ │ Ollama Server│ │Validation Srv│
└───────┬───────┘ └──────┬───────┘ └──────┬───────┘
        │                │                │
        ▼                ▼                ▼
┌───────────────┐ ┌──────────────┐ ┌──────────────┐
│  OMOP Database│ │ LLM Services │ │External Agents│
└───────────────┘ └──────────────┘ └──────────────┘
```

- **SQL Server**: Handles database operations and schema retrieval
- **Ollama Server**: Manages LLM interactions for SQL generation and explanations
- **Validation Server**: Validates SQL against OMOP CDM rules
- **Agent Integration**: Implements A2A protocol for external expert systems
- **Orchestrator**: Coordinates all servers and exposes REST/A2A APIs

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL or DuckDB with OMOP CDM data
- Ollama (or other compatible LLM service)

## 🚀 Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/omcp.git
cd omcp
```

2. **Set up a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure the application**

Copy the sample config and edit as needed:

```bash
cp config/config.sample.json config/config.json
# Edit config/config.json with your database connection details
```

5. **Create required directories**

```bash
mkdir -p logs
```

## 🔧 Configuration

The `config/config.json` file contains all settings:

- Database connection strings
- LLM API endpoints and models
- MCP server configurations
- External agent integrations

Example configuration:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 5433
  },
  "database": {
    "connection_strings": {
      "default": "postgresql://user:pass@localhost:5432/db?options=-c%20search_path=omop_cdm"
    }
  },
  "ollama": {
    "api_url": "http://localhost:11434/api/generate",
    "default_model": "codellama-7b"
  }
}
```

## 💻 Usage

### Starting the Server

```bash
python -m orchestrator.main
```

This starts the orchestrator, which launches all MCP servers.

### REST API Endpoints

#### Convert Natural Language to Answer

```http
POST /api/query
Content-Type: application/json

{
  "question": "How many patients have diabetes?",
  "context": null
}
```

#### Generate SQL Only

```http
POST /api/sql
Content-Type: application/json

{
  "question": "List all patients over 65 with hypertension",
  "context": null
}
```

#### Validate and Refine SQL

```http
POST /api/validate
Content-Type: application/json

{
  "question": "SELECT * FROM person p JOIN condition_occurrence co ON p.person_id = co.person_id WHERE co.condition_concept_id = 201826"
}
```

### A2A Protocol Integration

Use the A2A protocol endpoint for agent-to-agent communication:

```http
POST /a2a
Content-Type: application/json

{
  "messages": [
    {
      "role": "user",
      "content": "Find all female patients with lung cancer",
      "metadata": {}
    }
  ]
}
```

### Command-line Usage

For direct Claude integration using MCP:

```bash
claude-cli --mcp-server "python -m mcp_servers.sql_server"
```

Logs are written to `logs/omcp.log` by default.

## 🤝 Integration with Claude and Other LLMs

OMCP can be integrated with Claude Desktop:

1. Install Claude Desktop
2. Configure it to use OMCP as an MCP server:

```json
{
  "mcpServers": {
    "omcp": {
      "command": "python",
      "args": ["-m", "mcp_servers.sql_server"]
    }
  }
}
```

## 🛠️ Development

### Project Structure

```
omcp/
├── mcp_servers/           # MCP server implementations
│   ├── sql_server.py      # Database operations
│   ├── ollama_server.py   # LLM interactions
│   ├── validation_server.py # SQL validation
│   └── agent_server.py    # External agent integration
├── orchestrator/          # Orchestration components
│   ├── a2a.py             # A2A protocol implementation
│   ├── mcp_client.py      # MCP client for server communication
│   └── main.py            # Main application & API endpoints
├── config/                # Configuration files
├── schemas/               # OMOP schemas and validation rules
└── tests/                 # Unit and integration tests
```

### Adding Features

To add a new feature:

1. Add the appropriate tool to the relevant MCP server
2. Update the orchestrator to use the new tool
3. Add tests for the new functionality
4. Update documentation

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Niko Möller-Grell(https://github.com/nikomoegre)
- Shihao Shenzhang (https://github.com/shen1802)
- Vishnu V Chandrabalan (https://github.com/vvcb)

## 🙏 Acknowledgments

- [OHDSI](https://www.ohdsi.org/) for the OMOP Common Data Model
- [Model Context Protocol](https://modelcontextprotocol.io/) for the MCP specification
- [Google A2A Protocol](https://google.github.io/A2A/) for the A2A protocol specification
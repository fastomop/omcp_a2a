from mcp.server.fastmcp import FastMCP
import httpx
from typing import Dict, Any, List, Tuple, Optional

# Initialize MCP server
mcp = FastMCP(name="OMOP LLM MCP Server")


# Get configuration
def get_config():
    from app.core.config import settings
    return {
        "api_url": settings.config["ollama"]["api_url"],
        "default_model": settings.config["ollama"]["default_model"]
    }


@mcp.tool(
    name="Generate_SQL",
    description="Generate SQL from natural language using an LLM"
)
async def generate_sql(prompt: str, schema: str, model_name: str = None, system_prompt: str = None) -> Tuple[
    str, float]:
    """Generate SQL from natural language using Ollama

    Args:
        prompt: Natural language query
        schema: Database schema information
        model_name: Optional model name override
        system_prompt: Optional system prompt override

    Returns:
        Tuple of (generated SQL query, confidence score)
    """
    config = get_config()

    # Set up the model name and options
    model = model_name or config["default_model"]

    # Default system prompt for SQL generation
    default_system = "You are an expert in SQL and healthcare data analysis, specifically working with the OMOP Common Data Model (CDM)."
    system_message = system_prompt or default_system

    # Build the prompt for SQL generation
    full_prompt = f"""
Given the following OMOP CDM schema:

{schema}

Convert the following natural language query into a valid SQL query that follows OMOP CDM best practices:

"{prompt}"

Return ONLY the SQL query without any additional text, explanations, or markdown formatting.
Follow these important guidelines for OMOP CDM:
1. Always join person table when querying patient-level data
2. Always join concept tables when filtering by medical concepts
3. Use appropriate date ranges for temporal queries
4. Remember that most clinical data is in condition_occurrence, drug_exposure, measurement, and observation tables
5. Make sure to handle NULL values appropriately
"""

    # Prepare the request for Ollama
    ollama_request = {
        "model": model,
        "prompt": full_prompt,
        "system": system_message,
        "stream": False
    }

    try:
        # Use httpx for async HTTP requests
        async with httpx.AsyncClient() as client:
            response = await client.post(config["api_url"], json=ollama_request, timeout=240)
            response.raise_for_status()

            # Extract the generated response
            result = response.json()
            sql_query = result["response"].strip()

            # Clean up the SQL (in case it's wrapped in markdown code blocks)
            if sql_query.startswith("```") and sql_query.endswith("```"):
                # Extract SQL from markdown code block
                sql_query = sql_query.split("```")[1]
                if sql_query.startswith("sql"):
                    sql_query = sql_query[3:].strip()

            # Default confidence value
            confidence = 0.9

            # Return tuple
            return sql_query, confidence

    except Exception as e:
        raise Exception(f"Failed to generate SQL: {str(e)}")


@mcp.tool(
    name="Generate_Explanation",
    description="Generate an explanation for an SQL query"
)
async def generate_explanation(sql_query: str, model_name: str = None) -> str:
    """Generate an explanation for the SQL query

    Args:
        sql_query: SQL query to explain
        model_name: Optional model name override

    Returns:
        Natural language explanation
    """
    config = get_config()
    model = model_name or config["default_model"]

    explanation_prompt = f"""
Explain what this SQL query does in simple terms, focusing on the healthcare insights it provides:

```sql
{sql_query}
```

Explain in 2-3 sentences what clinical question this query answers and how it uses the OMOP CDM structure.
"""

    explanation_request = {
        "model": model,
        "prompt": explanation_prompt,
        "system": "You are a healthcare analytics expert explaining SQL queries to clinicians.",
        "stream": False
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(config["api_url"], json=explanation_request, timeout=120)
            response.raise_for_status()
            return response.json()["response"].strip()
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


@mcp.tool(
    name="Generate_Answer",
    description="Generate a natural language answer based on query, SQL, and results"
)
async def generate_answer(question: str, sql_query: str, results: str) -> str:
    """Generate a natural language answer to the original question

    Args:
        question: Original natural language question
        sql_query: SQL query that was executed
        results: Results from the query execution (CSV format)

    Returns:
        Natural language answer
    """
    config = get_config()

    answer_prompt = f"""
Given the following:

Question: "{question}"

SQL Query:
```sql
{sql_query}
```

Query Results:
```
{results}
```

Generate a comprehensive natural language answer to the original question based on these results.
Explain the insights from the data in a way that would be understandable to healthcare professionals.
"""

    answer_request = {
        "model": config["default_model"],
        "prompt": answer_prompt,
        "system": "You are a healthcare analytics expert explaining query results to clinicians.",
        "stream": False
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(config["api_url"], json=answer_request, timeout=240)
            response.raise_for_status()
            return response.json()["response"].strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


@mcp.tool(
    name="List_Available_Models",
    description="List available LLM models from Ollama"
)
async def list_available_models() -> List[Dict[str, Any]]:
    """List available models from Ollama

    Returns:
        List of available models
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=10)
            response.raise_for_status()
            return response.json().get("models", [])
    except Exception as e:
        raise Exception(f"Error listing models: {str(e)}")


if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")
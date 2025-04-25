from mcp.server.fastmcp import FastMCP
import json
import re
import httpx
from typing import Dict, Any, List, Tuple, Optional

# Initialize MCP server
mcp = FastMCP(name="OMOP Validation MCP Server")


# Load validation rules
def _load_validation_rules() -> Dict[str, Any]:
    """Load validation rules from file"""
    try:
        from app.core.config import settings
        rules_path = settings.get_validation_rules_path()
        with open(rules_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {
            "required_tables": [],
            "required_joins": [],
            "concept_tables": []
        }


# Initialize validation rules
VALIDATION_RULES = _load_validation_rules()


@mcp.tool(
    name="Validate_SQL_Query",
    description="Validate a SQL query against OMOP CDM validation rules"
)
def validate_query(sql_query: str) -> Dict[str, Any]:
    """Validate a SQL query against OMOP CDM rules

    Args:
        sql_query: SQL query to validate

    Returns:
        Validation result with is_valid flag and list of issues
    """
    # Initialize validation result
    validation_result = {
        "is_valid": True,
        "issues": []
    }

    # Convert SQL to lowercase for case-insensitive checks
    sql_lower = sql_query.lower()

    # Check for prohibited operations
    prohibited_patterns = [
        r'\bdrop\s+table\b',
        r'\btruncate\s+table\b',
        r'\bdelete\s+from\b',
        r'\bupdate\s+\w+\s+set\b',
        r'\balter\s+table\b'
    ]

    for pattern in prohibited_patterns:
        if re.search(pattern, sql_lower):
            validation_result["is_valid"] = False
            validation_result["issues"].append(
                "Query contains prohibited operations (DROP, TRUNCATE, DELETE, UPDATE, ALTER)")
            break

    # Check for required tables
    required_tables = VALIDATION_RULES.get("required_tables", [])
    for table in required_tables:
        trigger = table["when"].lower()
        table_name = table["name"].lower()

        # Check if the trigger word is in the query but the required table is not
        if trigger in sql_lower and table_name not in sql_lower:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Missing required table {table['name']} when querying {table['when']}")

    # Check for required joins
    required_joins = VALIDATION_RULES.get("required_joins", [])
    for join in required_joins:
        table1 = join["table1"].lower()
        table2 = join["table2"].lower()
        condition = join["condition"].lower()

        # If both tables are in the query, check for the join condition
        if table1 in sql_lower and table2 in sql_lower:
            # Look for different variants of the join condition
            condition_variants = [
                condition,
                condition.replace(" = ", "="),
                condition.replace("=", " = "),
                # Add other common variants
            ]

            if not any(variant in sql_lower for variant in condition_variants):
                validation_result["is_valid"] = False
                validation_result["issues"].append(
                    f"Missing proper join condition between {join['table1']} and {join['table2']}")

    # Check for concept_id filters when using concept tables
    concept_tables = VALIDATION_RULES.get("concept_tables", [])
    for concept_table in concept_tables:
        table_name = concept_table.lower()
        if table_name in sql_lower and "concept_id" not in sql_lower:
            validation_result["issues"].append(f"Warning: Querying {concept_table} without concept_id filter")

    # Check for date range filters on temporal queries
    if any(term in sql_lower for term in ["date", "datetime", "time"]):
        if not any(term in sql_lower for term in ["between", ">", "<", ">=", "<="]):
            validation_result["issues"].append("Warning: Temporal query without date range filter")

    # Check for basic SQL syntax issues (unbalanced parentheses, missing quotes)
    if sql_query.count('(') != sql_query.count(')'):
        validation_result["is_valid"] = False
        validation_result["issues"].append("SQL syntax error: Unbalanced parentheses")

    # Check for unclosed quotes
    quote_chars = ["'", '"']
    for quote in quote_chars:
        if sql_query.count(quote) % 2 != 0:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"SQL syntax error: Unclosed {quote} quotes")

    return validation_result


@mcp.tool(
    name="External_Validator",
    description="Call an external validator agent for SQL validation"
)
async def agent_validation(sql_query: str) -> Dict[str, Any]:
    """Call the external validator agent for SQL syntax validation

    Args:
        sql_query: SQL query to validate

    Returns:
        Validation result from external validator
    """
    try:
        from app.core.config import settings
        validator_url = settings.config["agents"]["medical_validator"]["url"]
        timeout = settings.config["agents"]["medical_validator"]["timeout"]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                validator_url,
                json={"query": sql_query},
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {
            "is_valid": False,
            "issues": [f"External validation failed: {str(e)}"]
        }


@mcp.tool(
    name="Comprehensive_Validation",
    description="Perform both local and external validation of a SQL query"
)
async def comprehensive_validation(sql_query: str) -> Dict[str, Any]:
    """Perform both local and external validation of an SQL query

    Args:
        sql_query: SQL query to validate

    Returns:
        Combined validation result
    """
    # First do local validation
    local_result = validate_query(sql_query)

    # If local validation fails, don't bother with external validation
    if not local_result["is_valid"]:
        return local_result

    # If local validation passes, try external validation
    try:
        external_result = await agent_validation(sql_query)
        if not external_result.get("is_valid", True):
            local_result["is_valid"] = False
            local_result["issues"].extend(external_result.get("issues", []))
    except Exception as e:
        local_result["issues"].append(f"Warning: External validation unavailable: {str(e)}")

    return local_result


if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")
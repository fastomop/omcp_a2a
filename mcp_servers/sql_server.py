from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, List, Tuple, Optional
import json
import time
from sqlalchemy import create_engine, text

# Initialize MCP server
mcp = FastMCP(name="OMOP SQL MCP Server")

# Internal state
_db_engines = {}


def get_db_engine(connection_id: Optional[str] = None, connection_string: Optional[str] = None):
    """Get or create a database engine"""
    global _db_engines

    if connection_string:
        # Create a temporary engine for one-time use
        return create_engine(connection_string)

    # Use a connection from the config
    conn_id = connection_id or "default"

    if conn_id not in _db_engines:
        try:
            from app.core.config import settings
            conn_string = settings.get_db_connection_string(conn_id)
            _db_engines[conn_id] = create_engine(conn_string)
        except Exception as e:
            raise Exception(f"Failed to create engine for {conn_id}: {e}")

    return _db_engines[conn_id]


@mcp.tool(
    name="Execute_SQL_Query",
    description="Execute a SQL query against an OMOP database and return results"
)
def execute_query(query: str, connection_id: str = None, connection_string: str = None) -> str:
    """Execute a SQL query and return results in CSV format

    Args:
        query: SQL query to execute
        connection_id: Optional ID for predefined connection
        connection_string: Optional direct connection string

    Returns:
        Results as CSV string
    """
    start_time = time.time()

    try:
        engine = get_db_engine(connection_id, connection_string)

        with engine.connect() as connection:
            result = connection.execute(text(query))
            column_names = result.keys()

            # Convert rows to CSV
            csv_data = ",".join(column_names) + "\n"
            for row in result:
                csv_data += ",".join([str(val) if val is not None else "" for val in row]) + "\n"

            execution_time = time.time() - start_time
            return f"# Execution time: {execution_time:.3f} seconds\n{csv_data}"

    except Exception as e:
        return f"Error executing query: {str(e)}"


@mcp.tool(
    name="Test_Connection",
    description="Test if a database connection is valid"
)
def test_connection(connection_string: str) -> bool:
    """Test if a connection string is valid

    Args:
        connection_string: Database connection string to test

    Returns:
        True if connection is valid, False otherwise
    """
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        return False


@mcp.tool(
    name="Get_OMOP_Schema",
    description="Get the OMOP CDM schema information for prompting"
)
def get_omop_schema() -> str:
    """Load and format OMOP CDM schema for prompting

    Returns:
        Formatted schema text
    """
    try:
        from app.core.config import settings
        schema_path = settings.get_omop_schema_path()

        with open(schema_path, "r") as f:
            schema_data = json.load(f)

        # Format the schema for the prompt
        schema_text = "OMOP CDM Database Schema:\n\n"

        # Add main tables first
        core_tables = ["person", "visit_occurrence", "condition_occurrence", "drug_exposure", "measurement",
                       "observation"]

        # First add core tables for better context
        schema_text += "Core Tables:\n"
        for table_name in core_tables:
            table = next((t for t in schema_data["tables"] if t["name"] == table_name), None)
            if table:
                schema_text += f"Table: {table['name']} - {table.get('description', '')}\n"

                for col in table["columns"]:
                    col_desc = f"  - {col['name']} ({col['data_type']})"
                    if col.get("description"):
                        col_desc += f": {col['description']}"
                    schema_text += col_desc + "\n"

                schema_text += "\n"

        # Then add other tables
        schema_text += "Other Tables:\n"
        for table in schema_data["tables"]:
            if table["name"] not in core_tables:
                schema_text += f"Table: {table['name']} - {table.get('description', '')}\n"

                # Add only key columns for non-core tables
                key_columns = [c for c in table["columns"] if
                               c.get("is_key", False) or "_id" in c["name"] or "concept_id" in c["name"]]
                for col in key_columns:
                    col_desc = f"  - {col['name']} ({col['data_type']})"
                    if col.get("description"):
                        col_desc += f": {col['description']}"
                    schema_text += col_desc + "\n"

                schema_text += f"  - plus {len(table['columns']) - len(key_columns)} more columns\n\n"

        # Add common relationships
        schema_text += "\nKey Relationships:\n"
        for relation in schema_data.get("relationships", []):
            schema_text += f"- {relation['source_table']}.{relation['source_column']} -> {relation['target_table']}.{relation['target_column']}\n"

        return schema_text

    except Exception as e:
        return f"OMOP CDM Schema unavailable: {str(e)}"


if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")
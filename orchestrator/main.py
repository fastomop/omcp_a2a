import asyncio
import os
import subprocess
import logging
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, Request, HTTPException, status, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .a2a import A2AProtocol
from .mcp_client import MCPClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")


# Define response models
class Query(BaseModel):
    question: str
    context: Optional[str] = None


class RefinementInfo(BaseModel):
    original_sql: str
    was_refined: bool


class NaturalLanguageResponse(BaseModel):
    answer: str
    sql: str
    confidence: float
    refinement_info: Optional[RefinementInfo] = None


class SQLResult(BaseModel):
    sql: str
    result: Any
    execution_time: float
    refinement_info: Optional[RefinementInfo] = None


class ValidationResult(BaseModel):
    is_valid: bool
    issues: Optional[List[str]] = None
    refinement_attempted: bool = False
    refinement_successful: bool = False
    refined_sql: Optional[str] = None
    refined_issues: Optional[List[str]] = None


# Orchestrator class
class MCPOrchestrator:
    """Orchestrator for multiple MCP servers with A2A protocol"""

    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        self.a2a = A2AProtocol()

    async def start_servers(self):
        """Start all MCP servers"""
        # Define server configurations - adjust paths based on your project structure
        server_configs = [
            {
                "name": "sql",
                "script_path": "mcp_servers/sql_server.py"
            },
            {
                "name": "ollama",
                "script_path": "mcp_servers/ollama_server.py"
            },
            {
                "name": "validation",
                "script_path": "mcp_servers/validation_server.py"
            },
            {
                "name": "agent",
                "script_path": "mcp_servers/agent_server.py"
            }
        ]

        # Start each server
        for config in server_configs:
            await self.start_server(config["name"], config["script_path"])

    async def start_server(self, name: str, script_path: str):
        """Start a single MCP server"""
        try:
            # Make script path absolute based on project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            abs_script_path = os.path.join(base_dir, script_path)

            logger.info(f"Starting MCP server: {name} with script {abs_script_path}")

            # Start the server process
            process = subprocess.Popen(
                ["python", abs_script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # Line buffered
                universal_newlines=False  # Use bytes for stdin/stdout
            )

            # Create client and start processing
            client = MCPClient(name, process)
            self.clients[name] = client
            await client.start()

            logger.info(f"Started MCP server: {name}")
        except Exception as e:
            logger.error(f"Failed to start MCP server {name}: {e}")

    async def stop_servers(self):
        """Stop all MCP servers"""
        for name, client in self.clients.items():
            await client.stop()
            logger.info(f"Stopped MCP server: {name}")
        self.clients = {}

    async def process_natural_language_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Process a natural language query using the orchestrated MCP servers"""
        try:
            # Step 1: Get schema if context not provided
            if not context:
                logger.info("Getting OMOP schema")
                context = await self.clients["sql"].call_tool("Get_OMOP_Schema", {})

            # Step 2: Generate SQL
            logger.info(f"Generating SQL for query: {query}")
            sql_result = await self.clients["ollama"].call_tool("Generate_SQL", {
                "prompt": query,
                "schema": context
            })

            if not sql_result:
                raise Exception("Failed to generate SQL")

            # Parse SQL result (tuple of SQL query and confidence)
            sql_query, confidence = sql_result
            logger.info(f"Generated SQL: {sql_query}")

            # Step 3: Validate and refine SQL if needed
            logger.info("Validating SQL")
            validation_result = await self.clients["validation"].call_tool("Comprehensive_Validation", {
                "sql_query": sql_query
            })

            # Store original SQL for refinement info
            original_sql = sql_query

            # Attempt refinement if validation failed
            refinement_attempted = False
            refinement_successful = False

            if not validation_result["is_valid"]:
                logger.info("Validation failed, attempting refinement")
                refinement_attempted = True

                # Call refinement tool if we implemented it
                if "Refine_SQL" in self.clients["validation"].available_tools:
                    refined_result = await self.clients["validation"].call_tool("Refine_SQL", {
                        "sql_query": sql_query,
                        "issues": validation_result["issues"]
                    })

                    if refined_result and refined_result.get("is_valid", False):
                        logger.info("Refinement successful")
                        sql_query = refined_result["refined_sql"]
                        validation_result = refined_result
                        refinement_successful = True
                    else:
                        logger.info("Refinement failed")

                # If validation still fails and no refinement, return error
                if not refinement_successful and not validation_result["is_valid"]:
                    return {
                        "error": "SQL validation failed",
                        "validation": validation_result,
                        "sql": sql_query,
                        "refinement_info": {
                            "original_sql": original_sql,
                            "was_refined": refinement_attempted
                        }
                    }

            # Step 4: Execute SQL
            logger.info("Executing SQL")
            query_result = await self.clients["sql"].call_tool("Execute_SQL_Query", {
                "query": sql_query
            })

            if not query_result:
                raise Exception("Failed to execute SQL query")

            # Step 5: Generate answer
            logger.info("Generating answer")
            answer = await self.clients["ollama"].call_tool("Generate_Answer", {
                "question": query,
                "sql_query": sql_query,
                "results": query_result
            })

            if not answer:
                raise Exception("Failed to generate answer")

            # Step 6: Get agent insights (optional)
            agent_insights = None
            try:
                logger.info("Getting agent insights")
                agent_insights = await self.clients["agent"].call_tool("Get_Agent_Insights", {
                    "prompt": query,
                    "sql": sql_query,
                    "agent_type": "medical_expert"
                })
            except Exception as e:
                logger.warning(f"Failed to get agent insights: {e}")

            return {
                "answer": answer,
                "sql": sql_query,
                "confidence": confidence,
                "validation": validation_result,
                "results": query_result,
                "agent_insights": agent_insights,
                "refinement_info": {
                    "original_sql": original_sql,
                    "was_refined": refinement_attempted and refinement_successful
                }
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"error": str(e)}


# FastAPI application
app = FastAPI(title="OMCP API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create router
router = APIRouter(prefix="/api")

# Create orchestrator instance
orchestrator = MCPOrchestrator()


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting OMCP Orchestrator")
    await orchestrator.start_servers()


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down OMCP Orchestrator")
    await orchestrator.stop_servers()


# A2A endpoint
@app.post("/a2a")
async def a2a_endpoint(request: Request):
    """A2A protocol endpoint"""
    try:
        body = await request.json()
        messages = body.get("messages", [])

        # Extract the first user message
        user_message = next((m for m in messages if m["role"] == "user"), None)
        if not user_message:
            return {"error": "No user message found"}

        query = user_message["content"]
        context = user_message.get("metadata", {}).get("context")

        logger.info(f"A2A request received: {query[:50]}...")

        # Process the query
        result = await orchestrator.process_natural_language_query(query, context)

        # Format as A2A response
        if "error" in result:
            response_message = {
                "role": "assistant",
                "content": f"Error: {result['error']}",
                "metadata": result
            }
        else:
            response_message = {
                "role": "assistant",
                "content": result["answer"],
                "metadata": {
                    "sql": result["sql"],
                    "confidence": result["confidence"],
                    "validation": result["validation"],
                    "agent_insights": result.get("agent_insights"),
                    "refinement_info": result.get("refinement_info")
                }
            }

        return {
            "messages": [response_message]
        }
    except Exception as e:
        logger.error(f"Error in A2A endpoint: {e}")
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Error processing request: {str(e)}"
                }
            ]
        }


# Query endpoint
@router.post("/query", response_model=NaturalLanguageResponse)
async def process_query(query: Query):
    """Process a natural language query about the OMOP CDM database"""
    logger.info(f"Received query: {query.question}")

    try:
        # Use the orchestrator to process the query
        result = await orchestrator.process_natural_language_query(query.question, query.context)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        # Create refinement info
        refinement_info = None
        if "refinement_info" in result:
            refinement_info = RefinementInfo(
                original_sql=result["refinement_info"]["original_sql"],
                was_refined=result["refinement_info"]["was_refined"]
            )

        return NaturalLanguageResponse(
            answer=result["answer"],
            sql=result["sql"],
            confidence=result["confidence"],
            refinement_info=refinement_info
        )

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# SQL endpoint
@router.post("/sql", response_model=SQLResult)
async def execute_sql(query: Query):
    """Generate and execute SQL from a natural language query"""
    logger.info(f"Generating SQL for: {query.question}")

    try:
        # Use the orchestrator to process the query
        result = await orchestrator.process_natural_language_query(query.question, query.context)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        # Create refinement info
        refinement_info = None
        if "refinement_info" in result:
            refinement_info = RefinementInfo(
                original_sql=result["refinement_info"]["original_sql"],
                was_refined=result["refinement_info"]["was_refined"]
            )

        return SQLResult(
            sql=result["sql"],
            result=result["results"],
            execution_time=0.0,  # We don't have this from the orchestrator yet
            refinement_info=refinement_info
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error executing SQL: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Validation endpoint
@router.post("/validate", response_model=ValidationResult)
async def validate_and_refine_sql(query: Query):
    """Validate SQL and attempt refinement if needed"""
    logger.info(f"Validating SQL: {query.question}")

    try:
        # If context contains SQL, use that, otherwise assume question is the SQL
        sql_query = query.context if query.context and query.context.strip().lower().startswith(
            "select") else query.question

        # Perform validation
        validation_result = await orchestrator.clients["validation"].call_tool("Comprehensive_Validation", {
            "sql_query": sql_query
        })

        # Attempt refinement if validation failed
        refinement_attempted = False
        refinement_successful = False
        refined_sql = None
        refined_issues = None

        if not validation_result["is_valid"]:
            refinement_attempted = True

            # Call refinement tool if we implemented it
            if "Refine_SQL" in orchestrator.clients["validation"].available_tools:
                refined_result = await orchestrator.clients["validation"].call_tool("Refine_SQL", {
                    "sql_query": sql_query,
                    "issues": validation_result["issues"]
                })

                if refined_result and refined_result.get("is_valid", False):
                    refined_sql = refined_result["refined_sql"]
                    refined_issues = refined_result.get("issues", [])
                    refinement_successful = True

        return ValidationResult(
            is_valid=validation_result["is_valid"],
            issues=validation_result["issues"],
            refinement_attempted=refinement_attempted,
            refinement_successful=refinement_successful,
            refined_sql=refined_sql,
            refined_issues=refined_issues
        )
    except Exception as e:
        logger.error(f"Error validating SQL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "servers": list(orchestrator.clients.keys())
    }


# Include the router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
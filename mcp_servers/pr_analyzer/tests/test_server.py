# test_server.py
import asyncio
import json
import pytest
import sys
import os
import shutil
import logging
from pathlib import Path
from fastmcp import FastMCP, Client
from mcp_servers.pr_analyzer.server import server as pr_analyzer_server_instance

# Configure logging for the test
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_serializable_content(obj):
    """Extract serializable content from a potentially complex response object."""
    if isinstance(obj, dict):
        if "type" in obj and obj.get("type") == "text" and "text" in obj:
            # Try to parse JSON text if it looks like JSON
            try:
                text = obj["text"]
                if isinstance(text, str) and (text.startswith("{") or text.startswith("[")):
                    return json.loads(text)
                return text
            except json.JSONDecodeError:
                return obj["text"]
        # Otherwise return the dict with processed values
        return {k: extract_serializable_content(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [extract_serializable_content(item) for item in obj]
    # For other types, return as is if it can be serialized, or as string otherwise
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)

@pytest.fixture
def mcp_server():
    """Pytest fixture to return the PR Analyzer MCP server instance."""
    logger.info("Setting up mcp_server fixture")
    
    # Extract tool names from the server for easier checking
    available_tool_objects = pr_analyzer_server_instance.list_tools()
    tool_names = [tool.name for tool in available_tool_objects if hasattr(tool, 'name')]
    logger.info(f"Available tool names: {tool_names}")
    
    # Print server details for debugging
    logger.info(f"Server instance ID: {id(pr_analyzer_server_instance)}")
    logger.info(f"Server name: {pr_analyzer_server_instance.name}")
    logger.info(f"Server instructions: {pr_analyzer_server_instance.instructions}")
    
    # Check for specific tools
    if not tool_names:
        logger.warning("No tools are available in the server!")
    elif "analyze_repository" in tool_names:
        logger.info("Tool 'analyze_repository' is available")
    elif "analysis.analyze_repository" in tool_names:
        logger.info("Tool 'analysis.analyze_repository' is available")
    else:
        logger.warning("Neither 'analyze_repository' nor 'analysis.analyze_repository' tool is available in the server!")
        
    # Try to get detailed information about registered tools
    try:
        if hasattr(pr_analyzer_server_instance, '_tools'):
            logger.info(f"Number of registered tools: {len(pr_analyzer_server_instance._tools)}")
            for i, tool in enumerate(pr_analyzer_server_instance._tools):
                logger.info(f"Tool {i}: name={tool.name}, async={getattr(tool, 'is_async', 'unknown')}")
    except Exception as e:
        logger.error(f"Error getting tool details: {e}")
    
    return pr_analyzer_server_instance

@pytest.mark.asyncio
async def test_server(mcp_server: FastMCP):
    """Test the PR Analyzer MCP server."""
    logger.info("Starting test_server test")
    
    # Log server information
    logger.info(f"Server name: {mcp_server.name}")
    tool_objects = mcp_server.list_tools()
    tool_names = [tool.name for tool in tool_objects if hasattr(tool, 'name')]
    logger.info(f"Available tool names: {tool_names}")
    
    # Set up a valid local Git repository for testing
    repo_path = "/Users/mg/mg-work/manav/work/ai-experiments/rag_modulo"
    if not Path(repo_path).exists() or not Path(repo_path).joinpath(".git").exists():
        logger.error(f"Test repository not found at {repo_path}")
        raise ValueError(f"Test repository not found at {repo_path}. Please provide a valid Git repository path.")
    
    logger.info(f"Using test repository at: {repo_path}")
    
    try:
        logger.info("Creating client and connecting to server")
        async with Client(mcp_server) as client:
            # Try both tool name formats
            tool_names_to_try = ["analyze_repository", "analysis.analyze_repository"]
            result = None
            
            # Check if the tool names are in the available tools
            for tool_name in tool_names_to_try:
                if tool_name in tool_names:
                    logger.info(f"Tool '{tool_name}' is available in the server")
                else:
                    logger.warning(f"Tool '{tool_name}' is NOT available in the server")
            
            # Try calling each tool
            for tool_name in tool_names_to_try:
                try:
                    logger.info(f"Attempting to call tool: {tool_name}")
                    logger.info(f"Environment variables: OPENAI_API_KEY present: {'OPENAI_API_KEY' in os.environ}")
                    
                    # Call the tool without a timeout
                    result = await client.call_tool(
                        tool_name,
                        {
                            "repo_path": repo_path,
                            "max_files": 10,  # Use smaller value for faster testing
                            "verbose": 2,
                        }
                    )
                    
                    logger.info(f"Tool call to {tool_name} completed successfully")
                    break  # Exit the loop if successful
                except Exception as e:
                    logger.warning(f"Failed to call tool {tool_name}: {e}")
                    logger.exception(e)  # Log the full stack trace
                    # Continue to try the next tool name
            
            if result:
                logger.info("Analysis result received")
                
                # Log the raw result structure for debugging
                logger.info(f"Raw result type: {type(result)}")
                logger.info(f"Raw result: {result}")
                
                # Process the result to extract serializable content
                try:
                    # Extract content from the potentially complex response
                    processed_result = extract_serializable_content(result)
                    
                    # Check for CrewAI-specific indicators
                    crewai_indicators = [
                        "Running crewai_approach/run_crew_pr.py",
                        "Analysis completed successfully",
                        "PR Recommendation",
                        "CrewAI"
                    ]
                    
                    result_str = str(result)
                    for indicator in crewai_indicators:
                        if indicator in result_str:
                            logger.info(f"CrewAI indicator found: '{indicator}'")
                        else:
                            logger.warning(f"CrewAI indicator NOT found: '{indicator}'")
                    
                    # Check for specific output patterns
                    output_file_found = "OUTPUT_FILE_PATH" in result_str
                    logger.info(f"Output file path found in result: {output_file_found}")
                    
                    # Print a summary of the result
                    if processed_result:
                        print("Analysis result (processed):")
                        # Try to pretty print the result, fall back to simple print if that fails
                        try:
                            result_json = json.dumps(processed_result, indent=2)
                            print(result_json)
                            
                            # Save the result to a file for inspection
                            with open("test_result.json", "w") as f:
                                f.write(result_json)
                            logger.info("Saved result to test_result.json")
                        except Exception as e:
                            logger.warning(f"Failed to JSON dump processed result: {e}")
                            print(str(processed_result))
                    else:
                        print("Empty or null result received")
                        
                except Exception as e:
                    logger.error(f"Error processing result: {e}", exc_info=True)
                    print("Raw result (could not process):", result)
            else:
                logger.error("All tool name attempts failed")
                raise Exception("All tool name attempts failed")
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_crewai_integration():
    """Test that the MCP server properly calls the CrewAI implementation."""
    logger.info("Starting CrewAI integration test")
    
    # Path to a valid git repository for testing
    test_repo_path = "/Users/mg/mg-work/manav/work/ai-experiments/rag_modulo"
    
    # Make sure the repo path exists
    if not Path(test_repo_path).exists() or not Path(test_repo_path).is_dir():
        logger.error(f"Test repository not found at {test_repo_path}")
        pytest.skip(f"Test repository not found at {test_repo_path}")
    
    # Check for OPENAI_API_KEY environment variable
    if "OPENAI_API_KEY" not in os.environ:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        # You might want to skip the test if no API key is available
        # pytest.skip("OPENAI_API_KEY not found in environment variables")
    
    logger.info(f"Using test repository: {test_repo_path}")
    
    # Initialize the MCP server
    try:
        async with Client(pr_analyzer_server_instance) as client:
            logger.info("Client connected to server")
            
            # Call the analyze_repository tool without a timeout
            try:
                result = await client.call_tool(
                    "analyze_repository",
                    {
                        "repo_path": test_repo_path,
                        "max_files": 10,  # Lower for faster testing
                        "verbose": 2,
                    }
                )
                
                logger.info("Tool call completed successfully")
                
                # Convert result to string for searching
                result_str = str(result)
                
                # Look for indicators that CrewAI was called
                crewai_indicators = [
                    "Running crewai_approach/run_crew_pr.py",
                    "Analysis completed successfully"
                ]
                
                for indicator in crewai_indicators:
                    if indicator in result_str:
                        logger.info(f"CrewAI indicator found: '{indicator}'")
                    else:
                        logger.warning(f"CrewAI indicator NOT found: '{indicator}'")
                
                # Check for expected output components
                assert "Running analysis for" in result_str, "CrewAI script was not called"
                
                # Log the result for inspection
                logger.info(f"Result type: {type(result)}")
                if isinstance(result, list):
                    logger.info(f"Result length: {len(result)}")
                    for i, item in enumerate(result[:5]):  # Show first 5 items
                        logger.info(f"Item {i}: {item}")
                
                # Save the result to a file
                with open("crewai_integration_result.txt", "w") as f:
                    f.write(str(result))
                logger.info("Saved result to crewai_integration_result.txt")
                
            except Exception as e:
                logger.error(f"Error calling tool: {e}", exc_info=True)
                pytest.fail(f"Error calling tool: {str(e)}")
    except Exception as e:
        logger.error(f"Error connecting to server: {e}", exc_info=True)
        pytest.fail(f"Error connecting to server: {str(e)}")

if __name__ == "__main__":
    logger.info("Running test_server directly")
    asyncio.run(test_server(pr_analyzer_server_instance))

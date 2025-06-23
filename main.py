#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server with FastAPI
A complete MCP server implementation supporting HTTP transport
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Protocol Models
class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class MCPNotification(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None

# Tool Schema Models
class ToolSchema(BaseModel):
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)

class Tool(BaseModel):
    name: str
    description: str
    inputSchema: ToolSchema

# Server Implementation
class MCPServer:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Any] = {}
        self.prompts: Dict[str, Any] = {}
        self._setup_default_tools()
    
    def _setup_default_tools(self):
        """Setup default tools for demonstration"""
        
        # Weather tool
        weather_tool = Tool(
            name="get_weather",
            description="Get current weather information for a location",
            inputSchema=ToolSchema(
                type="object",
                properties={
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                        "default": "fahrenheit"
                    }
                },
                required=["location"]
            )
        )
        self.tools["get_weather"] = weather_tool
        
        # Calculator tool
        calc_tool = Tool(
            name="calculate",
            description="Perform basic mathematical calculations",
            inputSchema=ToolSchema(
                type="object",
                properties={
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                    }
                },
                required=["expression"]
            )
        )
        self.tools["calculate"] = calc_tool
        
        # Time tool
        time_tool = Tool(
            name="get_current_time",
            description="Get the current date and time",
            inputSchema=ToolSchema(
                type="object",
                properties={
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (optional, defaults to UTC)",
                        "default": "UTC"
                    },
                    "format": {
                        "type": "string",
                        "description": "Date format (optional, defaults to ISO format)",
                        "default": "iso"
                    }
                },
                required=[]
            )
        )
        self.tools["get_current_time"] = time_tool
    
    async def handle_initialize(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": True
                },
                "resources": {
                    "subscribe": True,
                    "listChanged": True
                },
                "prompts": {
                    "listChanged": True
                }
            },
            "serverInfo": {
                "name": "python-mcp-server",
                "version": "1.0.0",
                "description": "A Python MCP server with multiple tools"
            }
        }
    
    async def handle_tools_list(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle tools/list request"""
        tools_list = []
        for tool in self.tools.values():
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema.model_dump()
            })
        
        return {"tools": tools_list}
    
    async def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Execute the tool
        if tool_name == "get_weather":
            return await self._execute_weather_tool(arguments)
        elif tool_name == "calculate":
            return await self._execute_calculator_tool(arguments)
        elif tool_name == "get_current_time":
            return await self._execute_time_tool(arguments)
        else:
            raise ValueError(f"Tool execution not implemented: {tool_name}")
    
    async def _execute_weather_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute weather tool"""
        location = args.get("location")
        unit = args.get("unit", "fahrenheit")
        
        # Mock weather data (in a real implementation, you'd call a weather API)
        weather_data = {
            "location": location,
            "temperature": 72 if unit == "fahrenheit" else 22,
            "unit": unit,
            "condition": "Partly cloudy",
            "humidity": 65,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Weather for {location}: {weather_data['temperature']}°{unit[0].upper()}, {weather_data['condition']}, Humidity: {weather_data['humidity']}%"
                }
            ],
            "isError": False
        }
    
    async def _execute_calculator_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute calculator tool"""
        expression = args.get("expression", "")
        
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set('0123456789+-*/().')
            if not all(c in allowed_chars or c.isspace() for c in expression):
                raise ValueError("Invalid characters in expression")
            
            result = eval(expression)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"{expression} = {result}"
                    }
                ],
                "isError": False
            }
        except Exception as e:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error calculating '{expression}': {str(e)}"
                    }
                ],
                "isError": True
            }
    
    async def _execute_time_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute time tool"""
        timezone = args.get("timezone", "UTC")
        format_type = args.get("format", "iso")
        
        current_time = datetime.now()
        
        if format_type == "iso":
            time_str = current_time.isoformat()
        else:
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Current time ({timezone}): {time_str}"
                }
            ],
            "isError": False
        }
    
    async def handle_resources_list(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle resources/list request"""
        return {"resources": []}
    
    async def handle_prompts_list(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle prompts/list request"""
        return {"prompts": []}
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCP request and return response"""
        method = request_data.get("method")
        params = request_data.get("params")
        request_id = request_data.get("id")
        
        try:
            if method == "initialize":
                result = await self.handle_initialize(params)
            elif method == "tools/list":
                result = await self.handle_tools_list(params)
            elif method == "tools/call":
                result = await self.handle_tools_call(params)
            elif method == "resources/list":
                result = await self.handle_resources_list(params)
            elif method == "prompts/list":
                result = await self.handle_prompts_list(params)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Error processing request {method}: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            }

# Global server instance
mcp_server = MCPServer()

# FastAPI setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI"""
    logger.info("Starting MCP Server...")
    yield
    logger.info("Shutting down MCP Server...")

app = FastAPI(
    title="Python MCP Server",
    description="Model Context Protocol Server with Python and FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Python MCP Server",
        "version": "1.0.0",
        "protocol": "Model Context Protocol",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tools_count": len(mcp_server.tools)
    }

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main MCP endpoint for handling protocol requests"""
    try:
        # Parse request body
        body = await request.body()
        request_data = json.loads(body.decode('utf-8'))
        
        logger.info(f"Received MCP request: {request_data.get('method', 'unknown')}")
        
        # Process the MCP request
        response_data = await mcp_server.process_request(request_data)
        
        logger.info(f"Sending MCP response for: {request_data.get('method', 'unknown')}")
        
        return JSONResponse(content=response_data)
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": "Parse error",
                    "data": str(e)
                }
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            }
        )

@app.get("/tools")
async def list_tools():
    """List available tools (convenience endpoint)"""
    tools_result = await mcp_server.handle_tools_list()
    return tools_result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Python MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger.info(f"Starting MCP Server on {args.host}:{args.port}")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False
    )
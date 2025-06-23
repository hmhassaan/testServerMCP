from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import asyncio
from datetime import datetime
import random
import math

app = FastAPI(title="MCP Test Server", version="1.0.0")

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP Protocol Models
class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class Tool(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]

class Resource(BaseModel):
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None

# Example tools configuration
TOOLS = [
    Tool(
        name="echo",
        description="Echo back the input text with a timestamp",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to echo back"
                }
            },
            "required": ["text"]
        }
    ),
    Tool(
        name="calculate",
        description="Perform basic mathematical calculations",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        }
    ),
    Tool(
        name="random_number",
        description="Generate a random number within a specified range",
        inputSchema={
            "type": "object",
            "properties": {
                "min": {
                    "type": "number",
                    "description": "Minimum value (default: 1)"
                },
                "max": {
                    "type": "number",
                    "description": "Maximum value (default: 100)"
                }
            }
        }
    ),
    Tool(
        name="get_time",
        description="Get current date and time information",
        inputSchema={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Time format: 'iso', 'timestamp', or 'readable' (default: 'readable')"
                }
            }
        }
    ),
    Tool(
        name="word_count",
        description="Count words, characters, and lines in text",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to analyze"
                }
            },
            "required": ["text"]
        }
    )
]

RESOURCES = [
    Resource(
        uri="memory://server-info",
        name="Server Information",
        description="Information about this MCP test server",
        mimeType="application/json"
    ),
    Resource(
        uri="memory://available-tools",
        name="Available Tools",
        description="List of all available tools on this server",
        mimeType="application/json"
    )
]

# Tool implementations
async def execute_echo(params: Dict[str, Any]) -> Dict[str, Any]:
    text = params.get("text", "")
    timestamp = datetime.now().isoformat()
    return {
        "echoed_text": text,
        "timestamp": timestamp,
        "message": f"Echo at {timestamp}: {text}"
    }

async def execute_calculate(params: Dict[str, Any]) -> Dict[str, Any]:
    expression = params.get("expression", "")
    try:
        # Safe evaluation of mathematical expressions
        allowed_chars = set("0123456789+-*/.()")
        if not all(c in allowed_chars or c.isdigit() or c.isspace() for c in expression):
            raise ValueError("Invalid characters in expression")
        
        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__
        }
    except Exception as e:
        return {
            "error": f"Calculation failed: {str(e)}",
            "expression": expression
        }

async def execute_random_number(params: Dict[str, Any]) -> Dict[str, Any]:
    min_val = params.get("min", 1)
    max_val = params.get("max", 100)
    
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    
    number = random.randint(int(min_val), int(max_val))
    return {
        "random_number": number,
        "range": f"{min_val} to {max_val}",
        "timestamp": datetime.now().isoformat()
    }

async def execute_get_time(params: Dict[str, Any]) -> Dict[str, Any]:
    format_type = params.get("format", "readable")
    now = datetime.now()
    
    if format_type == "iso":
        return {"time": now.isoformat(), "format": "ISO 8601"}
    elif format_type == "timestamp":
        return {"time": now.timestamp(), "format": "Unix timestamp"}
    else:  # readable
        return {"time": now.strftime("%Y-%m-%d %H:%M:%S"), "format": "Human readable"}

async def execute_word_count(params: Dict[str, Any]) -> Dict[str, Any]:
    text = params.get("text", "")
    
    words = len(text.split())
    characters = len(text)
    characters_no_spaces = len(text.replace(" ", ""))
    lines = len(text.split("\n"))
    
    return {
        "text_length": characters,
        "word_count": words,
        "character_count": characters,
        "character_count_no_spaces": characters_no_spaces,
        "line_count": lines,
        "average_words_per_line": round(words / max(lines, 1), 2)
    }

# Tool execution mapping
TOOL_EXECUTORS = {
    "echo": execute_echo,
    "calculate": execute_calculate,
    "random_number": execute_random_number,
    "get_time": execute_get_time,
    "word_count": execute_word_count
}

@app.get("/")
async def root():
    return {
        "message": "MCP Test Server Running!",
        "protocol": "Model Context Protocol",
        "version": "1.0.0",
        "tools_count": len(TOOLS),
        "resources_count": len(RESOURCES)
    }

@app.post("/mcp")
async def mcp_handler(request: MCPRequest):
    """Main MCP protocol handler - supports both regular HTTP and MCP client connections"""
    
    try:
        if request.method == "initialize":
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "logging": {}
                    },
                    "serverInfo": {
                        "name": "MCP Test Server",
                        "version": "1.0.0"
                    }
                }
            )
        
        elif request.method == "initialized":
            # Acknowledge initialization complete
            return MCPResponse(id=request.id, result={})
        
        elif request.method == "tools/list":
            return MCPResponse(
                id=request.id,
                result={
                    "tools": [tool.dict() for tool in TOOLS]
                }
            )
        
        elif request.method == "tools/call":
            params = request.params or {}
            tool_name = params.get("name")
            tool_arguments = params.get("arguments", {})
            
            if tool_name not in TOOL_EXECUTORS:
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found"
                    }
                )
            
            try:
                result = await TOOL_EXECUTORS[tool_name](tool_arguments)
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    }
                )
            except Exception as e:
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": -32603,
                        "message": f"Tool execution failed: {str(e)}"
                    }
                )
        
        elif request.method == "resources/list":
            return MCPResponse(
                id=request.id,
                result={
                    "resources": [resource.dict() for resource in RESOURCES]
                }
            )
        
        elif request.method == "resources/read":
            params = request.params or {}
            uri = params.get("uri")
            
            if uri == "memory://server-info":
                content = {
                    "server_name": "MCP Test Server",
                    "version": "1.0.0",
                    "protocol_version": "2024-11-05",
                    "tools_available": len(TOOLS),
                    "resources_available": len(RESOURCES),
                    "uptime": "Runtime information not tracked"
                }
                return MCPResponse(
                    id=request.id,
                    result={
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(content, indent=2)
                            }
                        ]
                    }
                )
            
            elif uri == "memory://available-tools":
                return MCPResponse(
                    id=request.id,
                    result={
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps([tool.dict() for tool in TOOLS], indent=2)
                            }
                        ]
                    }
                )
            
            else:
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Resource '{uri}' not found"
                    }
                )
        
        elif request.method == "ping":
            return MCPResponse(id=request.id, result={})
        
        else:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method '{request.method}' not implemented"
                }
            )
    
    except Exception as e:
        return MCPResponse(
            id=request.id,
            error={
                "code": -32603,
                "message": f"Internal server error: {str(e)}"
            }
        )

# Alternative endpoint that accepts raw JSON for MCP clients that don't use JSON-RPC wrapper
@app.post("/mcp/raw")
async def mcp_raw_handler(request: Request):
    """Raw MCP handler for clients that send unwrapped requests"""
    try:
        data = await request.json()
        
        # Wrap in MCP request format if not already wrapped
        if "jsonrpc" not in data:
            mcp_request = MCPRequest(
                method=data.get("method", ""),
                params=data.get("params"),
                id=data.get("id")
            )
        else:
            mcp_request = MCPRequest(**data)
        
        return await mcp_handler(mcp_request)
    
    except Exception as e:
        return MCPResponse(
            error={
                "code": -32700,
                "message": f"Parse error: {str(e)}"
            }
        )

# Legacy endpoint for backward compatibility
@app.post("/message")
async def receive_message(request: Request):
    data = await request.json()
    return {"status": "received", "data": data, "note": "Use /mcp endpoint for MCP protocol"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "protocol": "MCP",
        "tools": len(TOOLS),
        "resources": len(RESOURCES),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
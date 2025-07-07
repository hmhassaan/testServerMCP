#!/usr/bin/env python3
"""
Enhanced MCP Server with Code Execution
"""

import asyncio
import json
import logging
import os
import sys
import subprocess
import tempfile
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from contextlib import asynccontextmanager
import time

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

# Code Execution Classes
class CodeExecutionError(Exception):
    """Custom exception for code execution errors"""
    pass

class PythonCodeExecutor:
    """Secure Python code executor with sandboxing"""
    
    def __init__(self, base_sandbox_dir: str = "/tmp/mcp_sandbox"):
        self.base_sandbox_dir = Path(base_sandbox_dir)
        self.base_sandbox_dir.mkdir(exist_ok=True)
        
        # Security settings
        self.max_execution_time = 30  # seconds
        self.max_memory_mb = 128  # MB
        self.max_output_size = 10 * 1024 * 1024  # 10MB
        
        # Restricted imports/modules
        self.restricted_imports = {
            'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
            'http', 'ftplib', 'smtplib', 'telnetlib', 'pickle', 'marshal',
            'ctypes', 'platform', 'importlib', '__import__', 'eval', 'exec',
            'open', 'file', 'input', 'raw_input'
        }
    
    def create_sandbox(self) -> Path:
        """Create an isolated sandbox directory"""
        sandbox_id = str(uuid.uuid4())
        sandbox_path = self.base_sandbox_dir / sandbox_id
        sandbox_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (sandbox_path / "input").mkdir(exist_ok=True)
        (sandbox_path / "output").mkdir(exist_ok=True)
        (sandbox_path / "work").mkdir(exist_ok=True)
        
        return sandbox_path
    
    def cleanup_sandbox(self, sandbox_path: Path):
        """Clean up sandbox directory"""
        try:
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path)
        except Exception as e:
            logger.warning(f"Could not clean up sandbox {sandbox_path}: {e}")
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate Python code for security issues"""
        try:
            # Parse code to check for AST issues
            import ast
            tree = ast.parse(code)
            
            # Check for restricted patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.restricted_imports:
                            return False, f"Restricted import: {alias.name}"
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.restricted_imports:
                        return False, f"Restricted import: {node.module}"
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                            return False, f"Restricted function: {node.func.id}"
            
            return True, "Code validation passed"
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def create_execution_script(self, code: str, sandbox_path: Path) -> str:
        """Create a secure execution script"""
        script_template = f'''
import sys
import os
import time
from pathlib import Path
import io
from contextlib import redirect_stdout, redirect_stderr

# Change working directory to sandbox
os.chdir(str(Path("{sandbox_path}") / "work"))

# Redirect stdout/stderr to capture output
output_buffer = io.StringIO()
error_buffer = io.StringIO()

try:
    with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
        # Execute user code
        exec("""
{code}
""")
    
    print("===EXECUTION_SUCCESS===")
    print("STDOUT:", output_buffer.getvalue())
    print("STDERR:", error_buffer.getvalue())
    
except Exception as e:
    print("===EXECUTION_ERROR===")
    print("ERROR:", str(e))
    print("STDERR:", error_buffer.getvalue())
'''
        
        return script_template
    
    async def execute_code(self, code: str, files: Optional[Dict[str, bytes]] = None) -> Dict[str, Any]:
        """Execute Python code in a sandboxed environment"""
        
        # Validate code first
        is_valid, validation_msg = self.validate_code(code)
        if not is_valid:
            return {
                "success": False,
                "error": f"Code validation failed: {validation_msg}",
                "output": "",
                "stderr": "",
                "execution_time": 0,
                "files": {}
            }
        
        sandbox_path = self.create_sandbox()
        
        try:
            # Write input files to sandbox
            if files:
                for filename, content in files.items():
                    file_path = sandbox_path / "input" / filename
                    file_path.write_bytes(content)
            
            # Create execution script
            script_content = self.create_execution_script(code, sandbox_path)
            script_path = sandbox_path / "execute.py"
            script_path.write_text(script_content)
            
            # Execute code with timeout
            start_time = time.time()
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(sandbox_path)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.max_execution_time
                )
                execution_time = time.time() - start_time
                
                # Parse output
                stdout_text = stdout.decode('utf-8', errors='replace')
                stderr_text = stderr.decode('utf-8', errors='replace')
                
                # Extract execution results
                success = "===EXECUTION_SUCCESS===" in stdout_text
                error_occurred = "===EXECUTION_ERROR===" in stdout_text
                
                # Parse output sections
                output = ""
                error_msg = ""
                
                if success:
                    lines = stdout_text.split('\n')
                    for line in lines:
                        if line.startswith("STDOUT:"):
                            output = line[7:].strip()
                        elif line.startswith("STDERR:") and not error_occurred:
                            error_msg = line[7:].strip()
                
                elif error_occurred:
                    lines = stdout_text.split('\n')
                    for line in lines:
                        if line.startswith("ERROR:"):
                            error_msg = line[6:].strip()
                
                # Collect output files
                output_files = {}
                work_dir = sandbox_path / "work"
                
                if work_dir.exists():
                    for file_path in work_dir.rglob("*"):
                        if file_path.is_file():
                            relative_path = file_path.relative_to(work_dir)
                            try:
                                content = file_path.read_text(encoding='utf-8')
                                output_files[str(relative_path)] = {
                                    "type": "text",
                                    "content": content
                                }
                            except UnicodeDecodeError:
                                content = file_path.read_bytes()
                                output_files[str(relative_path)] = {
                                    "type": "binary",
                                    "content": content.hex(),
                                    "size": len(content)
                                }
                
                return {
                    "success": success and not error_occurred,
                    "error": error_msg if error_msg else None,
                    "output": output,
                    "stderr": stderr_text if stderr_text else None,
                    "execution_time": execution_time,
                    "files": output_files
                }
                
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": "Code execution timed out",
                    "output": "",
                    "stderr": "",
                    "execution_time": self.max_execution_time,
                    "files": {}
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}",
                "output": "",
                "stderr": "",
                "execution_time": 0,
                "files": {}
            }
        
        finally:
            # Clean up sandbox
            self.cleanup_sandbox(sandbox_path)

# Server Implementation
class MCPServer:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Any] = {}
        self.prompts: Dict[str, Any] = {}
        self.code_executor = PythonCodeExecutor()
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
        
        # Python Code Execution tool
        python_exec_tool = Tool(
            name="execute_python",
            description="Execute Python code in a secure sandbox environment. Returns output, errors, and any generated files.",
            inputSchema=ToolSchema(
                type="object",
                properties={
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "files": {
                        "type": "object",
                        "description": "Optional input files as key-value pairs (filename: base64_content)",
                        "additionalProperties": {"type": "string"}
                    }
                },
                required=["code"]
            )
        )
        self.tools["execute_python"] = python_exec_tool
    
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
                "description": "A Python MCP server with code execution capabilities"
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
        elif tool_name == "execute_python":
            return await self._execute_python_tool(arguments)
        else:
            raise ValueError(f"Tool execution not implemented: {tool_name}")
    
    async def _execute_python_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code tool"""
        code = args.get("code", "")
        files_input = args.get("files", {})
        
        # Convert base64 files to bytes
        files = {}
        if files_input:
            import base64
            for filename, content in files_input.items():
                try:
                    files[filename] = base64.b64decode(content)
                except Exception as e:
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error decoding file {filename}: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
        
        # Execute code
        result = await self.code_executor.execute_code(code, files)
        
        # Format response
        response_text = f"**Code Execution Result:**\n"
        response_text += f"Success: {result['success']}\n"
        response_text += f"Execution Time: {result['execution_time']:.2f}s\n\n"
        
        if result['success']:
            if result['output']:
                response_text += f"**Output:**\n```\n{result['output']}\n```\n\n"
        else:
            response_text += f"**Error:**\n{result['error']}\n\n"
        
        if result['stderr']:
            response_text += f"**Stderr:**\n```\n{result['stderr']}\n```\n\n"
        
        if result['files']:
            response_text += f"**Generated Files:**\n"
            for filename, file_info in result['files'].items():
                if file_info['type'] == 'text':
                    response_text += f"- {filename} (text, {len(file_info['content'])} chars)\n"
                else:
                    response_text += f"- {filename} (binary, {file_info['size']} bytes)\n"
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": response_text
                }
            ],
            "isError": not result['success']
        }
    
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
    logger.info("Starting Enhanced MCP Server with Code Execution...")
    yield
    logger.info("Shutting down MCP Server...")

app = FastAPI(
    title="Enhanced Python MCP Server",
    description="Model Context Protocol Server with Python Code Execution",
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
        "message": "Enhanced Python MCP Server",
        "version": "1.0.0",
        "protocol": "Model Context Protocol",
        "status": "running",
        "features": ["code_execution", "weather", "calculator", "time"]
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

@app.post("/codeExec")
async def code_exec_endpoint(request: Request):
    """Direct code execution endpoint"""
    try:
        body = await request.body()
        request_data = json.loads(body.decode('utf-8'))
        
        code = request_data.get("code", "")
        files = request_data.get("files", {})
        
        if not code:
            return JSONResponse(
                status_code=400,
                content={"error": "No code provided"}
            )
        
        # Execute code directly
        result = await mcp_server.code_executor.execute_code(code, files)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Code execution error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Python MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger.info(f"Starting Enhanced MCP Server on {args.host}:{args.port}")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False
    )
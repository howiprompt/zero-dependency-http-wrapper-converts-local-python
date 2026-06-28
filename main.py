"""
Zero-dependency HTTP wrapper that converts local Python functions into agent-callable API endpoints.

Proposed, voted, built and 2-agent-verified by the HowiPrompt autonomous agent guild.
Free and MIT-licensed. More agent-built tools: https://howiprompt.xyz
Why this exists: pewdiepie-archdaemon/odysseus requires a full self-hosted workspace setup; this is a single-file alternative for the 'lazy senior dev' (ponytail) who wants to instantly expose pre-existing code to an 
"""
#!/usr/bin/env python3
"""
PyAgentWrap: Zero-Dependency Local Function-to-API Bridge
========================================================

A production-ready CLI tool that converts a standard Python script into a local 
HTTP server, exposing its defined functions as callable API endpoints for AI agents.

Usage Examples:
--------------
1. Basic usage (serves on default localhost:8000):
   $ python pyagentwrap.py my_tools.py

2. Custom host and port:
   $ python pyagentwrap.py my_tools.py --host 0.0.0.0 --port 9000

3. Securing the endpoint with an API Key (set via ENV):
   $ export AGENT_API_KEY="super_secret_key"
   $ python pyagentwrap.py my_tools.py

4. Interacting with the API:
   # Get manifest
   $ curl http://localhost:8000/

   # Call function 'calculate_sum' defined in my_tools.py
   $ curl -X POST http://localhost:8000/calculate_sum \
         -H "Content-Type: application/json" \
         -d '{"x": 10, "y": 32}'
"""

import argparse
import importlib.util
import inspect
import json
import logging
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple, Type
from urllib.parse import parse_qs, urlparse

# Configure logging to stdout so the main process isn't cluttered
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pyagentwrap")


# =============================================================================
# JSON Serialization Utilities
# =============================================================================

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that handles non-standard Python types commonly 
    found in data processing scripts.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        if hasattr(obj, 'isoformat'):  # Handles datetime and date objects
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def jsonify(data: Any, status: int = 200) -> bytes:
    """
    Converts Python data to a JSON byte string with proper headers.
    """
    try:
        json_bytes = json.dumps(data, cls=EnhancedJSONEncoder, indent=2).encode('utf-8')
        return json_bytes
    except Exception as e:
        logger.error(f"JSON Serialization Error: {e}")
        error_payload = {"error": "Internal serialization error", "details": str(e)}
        return json.dumps(error_payload).encode('utf-8')


# =============================================================================
# Module Introspection & Manifest Generation
# =============================================================================

FunctionManifest = Dict[str, Any]

class IntrospectionError(Exception):
    """Raised when module introspection fails."""
    pass


def load_target_module(file_path: str) -> Any:
    """
    Dynamically loads a Python module from a given file path.
    
    Args:
        file_path: Absolute or relative path to the .py file.
        
    Returns:
        The loaded module object.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ImportErrror: If the file cannot be imported/parsed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target file not found: {file_path}")
    
    path_abs = os.path.abspath(file_path)
    module_name = os.path.splitext(os.path.basename(path_abs))[0]
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, path_abs)
        if spec is None or spec.loader is None:
            raise IntrospectionError(f"Could not create module spec for {file_path}")
            
        module = importlib.util.module_from_spec(spec)
        # Execute the module to load functions into namespace
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Failed to load module {file_path}: {e}")
        raise ImportError(f"Module import failed: {e}")


def get_type_hint_string(hint: Any) -> str:
    """
    Safely converts a type hint object to a readable string.
    """
    if hasattr(hint, '__name__'):
        return hint.__name__
    return str(hint).replace('typing.', '')


def generate_tool_manifest(module: Any) -> List[FunctionManifest]:
    """
    Inspects the loaded module to find public functions and generate 
    a OpenAI-compatible manifest structure.
    
    Args:
        module: The loaded Python module.
        
    Returns:
        A list of dictionaries describing available functions.
    """
    tools = []
    
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        # Skip private functions
        if name.startswith('_'):
            continue
            
        # Ensure function is defined in the target module, not imported
        if obj.__module__ != module.__name__:
            continue
            
        try:
            sig = inspect.signature(obj)
            doc = inspect.getdoc(obj) or "No description provided."
            
            parameters = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                param_type = "any"
                if param.annotation != inspect.Parameter.empty:
                    param_type = get_type_hint_string(param.annotation)
                
                parameters[param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}"
                }
                
                # Handle defaults
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
            
            tool_def = {
                "name": name,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required
                }
            }
            tools.append(tool_def)
            
        except Exception as e:
            logger.warning(f"Skipping function {name} due to introspection error: {e}")
            
    return tools


# =============================================================================
# HTTP Server Request Handler
# =============================================================================

class AgentRequestHandler(BaseHTTPRequestHandler):
    """
    Handles HTTP requests for function execution and manifest serving.
    """
    
    # The module to operate on is injected by the server startup process
    _target_module: Optional[Any] = None
    _api_key: Optional[str] = None
    
    def log_message(self, format: str, *args: Any) -> None:
        """Override to use standard logger instead of stderr."""
        logger.info(f"{self.address_string()} - {format % args}")

    def _validate_auth(self) -> bool:
        """
        Checks for API Key if one is configured in the environment.
        """
        if not self._api_key:
            return True
            
        provided_key = self.headers.get('X-API-Key') or self.headers.get('Authorization')
        # Handle "Bearer <token>" format
        if provided_key and provided_key.startswith('Bearer '):
            provided_key = provided_key[7:]
            
        if provided_key == self._api_key:
            return True
            
        logger.warning(f"Failed auth attempt from {self.address_string()}")
        self.send_response(403)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Forbidden", "message": "Invalid or missing API Key"}).encode())
        return False

    def _send_json_response(self, data: Any, status: int = 200) -> None:
        """Helper to send JSON responses with standard headers."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(jsonify(data))

    def do_GET(self) -> None:
        """Handles GET requests (Manifest)."""
        if not self._validate_auth():
            return

        if self.path == '/' or self.path == '/manifest':
            if not self._target_module:
                self._send_json_response({"error": "Server configuration error: No module loaded"}, 500)
                return
                
            tools = generate_tool_manifest(self._target_module)
            manifest = {
                "version": "1.0.0",
                "server": "PyAgentWrap",
                "tools": tools
            }
            self._send_json_response(manifest)
        else:
            self._send_json_response({"error": "Not Found", "message": "Only GET / is supported for manifests"}, 404)

    def do_POST(self) -> None:
        """Handles POST requests (Function Execution)."""
        if not self._validate_auth():
            return

        # Parse endpoint path (e.g., /my_function)
        parsed_path = urlparse(self.path).path
        function_name = parsed_path.strip('/')
        
        if not function_name:
            self._send_json_response({"error": "Bad Request", "message": "Function name missing in path"}, 400)
            return

        if not self._target_module:
            self._send_json_response({"error": "Server configuration error: No module loaded"}, 500)
            return

        # Check if function exists
        if not hasattr(self._target_module, function_name):
            self._send_json_response({
                "error": "Function not found", 
                "message": f"The function '{function_name}' is not available in the loaded module."
            }, 404)
            return

        func = getattr(self._target_module, function_name)
        if not callable(func):
            self._send_json_response({"error": "Invalid target", "message": "Target is not a callable function"}, 400)
            return

        # Read Payload
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            # Allow POST with empty body if function has no required args
            payload_bytes = b'{}'
        else:
            payload_bytes = self.rfile.read(content_length)

        try:
            payload = json.loads(payload_bytes.decode('utf-8'))
        except json.JSONDecodeError:
            self._send_json_response({"error": "Invalid JSON", "message": "Request body must be valid JSON"}, 400)
            return

        if not isinstance(payload, dict):
            self._send_json_response({"error": "Invalid JSON", "message": "Request body must be a JSON object (key-value pairs)"}, 400)
            return

        # Execute Function
        try:
            # Inspect signature to validate arguments or rely on Python's built-in errors
            # We pass only arguments requested by the function signature if possible
            sig = inspect.signature(func)
            
            # Filter arguments:只传递函数签名中存在的参数
            valid_kwargs = {}
            for param_name in sig.parameters:
                if param_name in payload:
                    valid_kwargs[param_name] = payload[param_name]
            
            # Execute
            logger.info(f"Executing function '{function_name}' with args: {valid_kwargs}")
            result = func(**valid_kwargs)
            
            response_data = {
                "function": function_name,
                "status": "success",
                "result": result
            }
            self._send_json_response(response_data)

        except TypeError as te:
            logger.error(f"Type error in {function_name}: {te}")
            self._send_json_response({
                "error": "Argument Error", 
                "message": str(te), 
                "hint": "Check function signature and provided JSON keys."
            }, 400)
        except Exception as e:
            logger.error(f"Runtime error in {function_name}: {e}\n{traceback.format_exc()}")
            self._send_json_response({
                "error": "Execution Error", 
                "message": str(e),
                "traceback": traceback.format_exc() if os.environ.get("DEBUG") else None
            }, 500)


# =============================================================================
# Main Application Entry Point
# =============================================================================

def run_server(file_path: str, host: str, port: int) -> None:
    """
    Initializes and runs the HTTP server wrapping the target file.
    """
    try:
        # 1. Load the Module
        logger.info(f"Loading module from: {file_path}")
        module = load_target_module(file_path)
        
        # 2. Inject state into Handler
        AgentRequestHandler._target_module = module
        AgentRequestHandler._api_key = os.environ.get("AGENT_API_KEY")
        
        if AgentRequestHandler._api_key:
            logger.info("API Key protection enabled (checking AGENT_API_KEY).")
        else:
            logger.info("No API Key set. Running in open mode.")

        # 3. Verify functionality
        tools = generate_tool_manifest(module)
        if not tools:
            logger.warning("No public functions found in the target script. The server will serve an empty manifest.")
        else:
            logger.info(f"Successfully loaded {len(tools)} callable function(s): {', '.join([t['name'] for t in tools])}")

        # 4. Start Server
        server_address = (host, port)
        httpd = HTTPServer(server_address, AgentRequestHandler)
        
        logger.info("-" * 50)
        logger.info(f"PyAgentWrap Server started successfully!")
        logger.info(f"Serving: http://{host}:{port}")
        logger.info(f"Manifest: http://{host}:{port}/")
        logger.info("Press Ctrl+C to stop the server.")
        logger.info("-" * 50)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("\nShutting down server...")
            httpd.shutdown()
            
    except FileNotFoundError as e:
        logger.error(f"Startup Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected Critical Error: {e}")
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """CLI Interface entry point."""
    parser = argparse.ArgumentParser(
        description="PyAgentWrap: Convert local Python functions into agent-callable API endpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python pyagentwrap.py ./my_script.py --port 8080"
    )
    
    parser.add_argument(
        "target_file",
        type=str,
        help="Path to the Python .py file containing functions to expose."
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)."
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)."
    )
    
    args = parser.parse_args()
    
    if not args.target_file.endswith('.py'):
        logger.warning("Target file does not end with .py, attempting to load anyway...")
        
    run_server(args.target_file, args.host, args.port)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
ToonDB MCP Chat Client - Test client for ToonDB MCP Server

This client:
1. Connects to the ToonDB MCP server via STDIO
2. Uses Azure OpenAI for chat
3. Allows the LLM to call ToonDB tools via MCP

Usage:
    python test_mcp_chat.py

Requirements:
    pip install openai
"""

import json
import subprocess
import sys
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# MCP Server Configuration
MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH")
MCP_DB_PATH = os.getenv("MCP_DB_PATH")


class MCPClient:
    """Simple MCP client that communicates via STDIO"""

    def __init__(self, server_cmd: list[str]):
        self.server_cmd = server_cmd
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.tools = []

    def start(self):
        """Start the MCP server process"""
        print(f"🚀 Starting MCP server: {' '.join(self.server_cmd)}")
        self.process = subprocess.Popen(
            self.server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # Binary mode for proper framing
        )
        print("✅ MCP server started")

    def stop(self):
        """Stop the MCP server process"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("🛑 MCP server stopped")

    def send_message(self, message: dict) -> dict:
        """Send a JSON-RPC message and receive response"""
        if not self.process:
            raise RuntimeError("MCP server not started")

        # Serialize message
        msg_bytes = json.dumps(message).encode("utf-8")

        # Send with Content-Length framing
        header = f"Content-Length: {len(msg_bytes)}\r\n\r\n"
        self.process.stdin.write(header.encode("utf-8"))
        self.process.stdin.write(msg_bytes)
        self.process.stdin.flush()

        # Read response header
        header_line = b""
        while True:
            byte = self.process.stdout.read(1)
            if not byte:
                raise RuntimeError("MCP server closed connection")
            header_line += byte
            if header_line.endswith(b"\r\n\r\n"):
                break

        # Parse Content-Length
        header_str = header_line.decode("utf-8")
        content_length = None
        for line in header_str.split("\r\n"):
            if line.startswith("Content-Length:"):
                content_length = int(line.split(":")[1].strip())
                break

        if content_length is None:
            raise RuntimeError(f"Missing Content-Length in response: {header_str}")

        # Read response body
        body = self.process.stdout.read(content_length)
        return json.loads(body.decode("utf-8"))

    def next_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id

    def initialize(self) -> dict:
        """Initialize MCP connection"""
        response = self.send_message({
            "jsonrpc": "2.0",
            "id": self.next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "toondb-test-client", "version": "1.0.0"}
            }
        })
        print(f"📋 Server: {response.get('result', {}).get('serverInfo', {})}")
        return response

    def list_tools(self) -> list:
        """Get list of available tools"""
        response = self.send_message({
            "jsonrpc": "2.0",
            "id": self.next_id(),
            "method": "tools/list",
            "params": {}
        })
        self.tools = response.get("result", {}).get("tools", [])
        return self.tools

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool"""
        response = self.send_message({
            "jsonrpc": "2.0",
            "id": self.next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        })
        return response


class ChatClient:
    """Chat client that uses Azure OpenAI with ToonDB MCP tools"""

    def __init__(self, mcp_client: MCPClient):
        self.mcp = mcp_client
        self.messages = []
        self.openai_client = None

    def setup_openai(self):
        """Initialize Azure OpenAI client"""
        try:
            from openai import AzureOpenAI
            self.openai_client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            print("✅ Azure OpenAI client initialized")
        except ImportError:
            print("❌ Please install openai: pip install openai")
            sys.exit(1)

    def get_openai_tools(self) -> list:
        """Convert MCP tools to OpenAI function format"""
        openai_tools = []
        for tool in self.mcp.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
                }
            })
        return openai_tools

    def chat(self, user_message: str) -> str:
        """Send a message and get response with tool calling"""
        self.messages.append({"role": "user", "content": user_message})

        # System message
        system_msg = {
            "role": "system",
            "content": """You are a helpful AI assistant with access to ToonDB - an AI-native database.
You can use the available tools to:
- Query the database (toondb.query)
- Get/Put/Delete data by path (toondb.get, toondb.put, toondb.delete)
- List tables (toondb.list_tables)
- Get context for AI workloads (toondb.context_query)

When the user asks about data, use the appropriate tools to help them."""
        }

        # Get available tools
        tools = self.get_openai_tools()

        # Call Azure OpenAI
        response = self.openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[system_msg] + self.messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
        )

        assistant_message = response.choices[0].message

        # Handle tool calls
        if assistant_message.tool_calls:
            # Add assistant message with tool calls
            self.messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"\n🔧 Calling tool: {tool_name}")
                print(f"   Arguments: {json.dumps(tool_args, indent=2)}")

                # Call MCP tool
                result = self.mcp.call_tool(tool_name, tool_args)

                # Extract result content
                if "result" in result:
                    content = result["result"].get("content", [])
                    if content and len(content) > 0:
                        tool_result = content[0].get("text", str(result))
                    else:
                        tool_result = str(result["result"])
                elif "error" in result:
                    tool_result = f"Error: {result['error']}"
                else:
                    tool_result = str(result)

                print(f"   Result: {tool_result[:200]}...")

                # Add tool response
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            # Get final response after tool calls
            response = self.openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[system_msg] + self.messages,
            )
            assistant_message = response.choices[0].message

        # Add final response to history
        self.messages.append({"role": "assistant", "content": assistant_message.content})

        return assistant_message.content


def main():
    """Main entry point"""
    print("=" * 60)
    print("🗄️  ToonDB MCP Chat Client")
    print("=" * 60)

    # Create MCP client
    mcp = MCPClient([MCP_SERVER_PATH, "--db", MCP_DB_PATH])

    try:
        # Start MCP server
        mcp.start()

        # Initialize
        mcp.initialize()

        # List tools
        tools = mcp.list_tools()
        print(f"\n📦 Available tools ({len(tools)}):")
        for tool in tools:
            print(f"   • {tool['name']}: {tool.get('description', '')[:60]}...")

        # Create chat client
        chat = ChatClient(mcp)
        chat.setup_openai()

        print("\n" + "=" * 60)
        print("💬 Chat with ToonDB (type 'quit' to exit)")
        print("=" * 60)

        # Interactive chat loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    break

                response = chat.chat(user_input)
                print(f"\n🤖 Assistant: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    finally:
        mcp.stop()


if __name__ == "__main__":
    main()

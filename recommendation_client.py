# ==========================================================================================================
# Imports
# ==========================================================================================================

import asyncio
import json
import sys

from MCPClient import MCPClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (Root, TextContent, CreateMessageResult, CreateMessageRequestParams)



# ==========================================================================================================
# Client declaration
# ==========================================================================================================

# Configuration
SERVER_SCRIPT = str("server.py")

# StdioServerParameters launches "python server.py" via stdin/stdout
server_params = StdioServerParameters(
    command="python",
    args=[SERVER_SCRIPT],
)



# ==========================================================================================================
# Helper functions
# ==========================================================================================================

async def call_tool(tool_name: str, arguments: dict) -> dict:
    """Connect to the server, call a tool, and return the parsed JSON result."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            # The server always returns a single TextContent item
            return json.loads(result.content[0].text)



# ==========================================================================================================
# Beginning of execution
# ==========================================================================================================

async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect(sys.argv[1])
        await client.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        await client.cleanup()



# ==========================================================================================================
# Entry point
# ==========================================================================================================

if __name__ == "__main__":
    asyncio.run(main())
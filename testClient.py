# test_client.py — a minimal MCP client that connects to your server and calls a tool
import asyncio
from fastmcp import Client

# Point the client at your server file. FastMCP infers how to launch it.
client = Client("server.py")

async def main():
    async with client:
        # 1. List the tools the server exposes
        tools = await client.list_tools()
        print("Available tools:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        # 2. Actually call one — list files in a public repo
        print("\nCalling list_repo_files on octocat/Hello-World...\n")
        result = await client.call_tool(
            "list_repo_files",
            {"repo": "Varshini0511/multi-agent-travel-planner"}
        )
        print(result.data)

asyncio.run(main())
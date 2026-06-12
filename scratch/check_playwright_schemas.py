import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_SERVERS = {
    "playwright": {
        "command": "npx",
        "args": ["-y", "@executeautomation/playwright-mcp-server"],
        "transport": "stdio",
        "env": {
            "PLAYWRIGHT_MCP_HEADLESS": "true"
        }
    }
}

async def main():
    client = MultiServerMCPClient(MCP_SERVERS, tool_name_prefix=True)
    tools = await client.get_tools()
    for t in tools:
        if "get_visible_text" in t.name or "navigate" in t.name:
            print(f"Tool Name: {t.name}")
            print(f"Description: {t.description}")
            print(f"Args Schema: {json.dumps(t.args, indent=2)}")
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())

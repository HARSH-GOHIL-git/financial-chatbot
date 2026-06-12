import asyncio
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
    
    navigate_tool = next(t for t in tools if t.name == "playwright_playwright_navigate")
    get_text_tool = next(t for t in tools if t.name == "playwright_playwright_get_visible_text")
    
    print("Navigating...")
    res_nav = await navigate_tool.ainvoke({"url": "https://example.com"})
    print("Navigate Result:", res_nav)
    
    print("\nGetting visible text...")
    res_text = await get_text_tool.ainvoke({})
    print("Get Text Result:", res_text)

if __name__ == "__main__":
    asyncio.run(main())

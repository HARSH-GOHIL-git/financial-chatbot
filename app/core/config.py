import os
from dotenv import load_dotenv

# Load .env from the project root (two levels above this file: app/core/ -> app/ -> root)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")

load_dotenv(env_path)

DB_URI = os.getenv("DB_URI")
MCP_FS_ROOT = os.path.abspath(os.getenv("MCP_FS_ROOT", os.path.expanduser("~")))

MCP_SERVERS = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            MCP_FS_ROOT,
        ],
        "transport": "stdio",
    },
    "playwright": {
        "command": "npx",
        "args": ["-y", "@executeautomation/playwright-mcp-server"],
        "transport": "stdio",
        "env": {
            "PLAYWRIGHT_MCP_HEADLESS": "true",
            **os.environ
        }
    },
}

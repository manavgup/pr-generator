# PR Generator MCP Server

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run tests:
   ```bash
   python -m pytest pr_analyzer/tests/
   ```

3. Test with MCP Inspector:
   ```bash
   mcp dev pr_analyzer/server.py
   ```

4. Configure Claude Desktop:
   - Copy `config/claude_desktop_template.json`
   - Update paths to absolute paths
   - Save as Claude Desktop config

## Next Steps

1. Import your existing CrewAI tools in `server.py`
2. Replace mock implementations with real tool calls
3. Add more tools as needed
4. Test with real repositories
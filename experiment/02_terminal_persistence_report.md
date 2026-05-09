# Experiment 02: Persistent Terminal Sessions via Streamable HTTP

## Objective
To validate if it is possible to create an MCP tool for a bash terminal on the host machine where terminal session persistence is tied to the MCP session ID when using Streamable HTTP transport.

## Methodology
1. **MCP Server**: Implemented a server using `fastmcp` with a `SessionTerminalRegistry`.
    - Maps `mcp-session-id` to a `TerminalExecutor` instance.
    - Uses `SessionCleanupMiddleware` to intercept `DELETE /mcp` requests for session cleanup.
    - Implemented a reaper loop to clean up idle sessions (default 300s).
2. **MCP Client (Agent)**: Used `openhands-sdk` to create an agent that connects to the MCP server.
    - Triggered sequential tool calls to verify state persistence (environment variables).
3. **Transport**: Configured for `streamable-http` to leverage session affinity.

## Key Findings
- **Session ID Extraction**: `fastmcp`'s `Context` provides `ctx.session_id` which automatically resolves the `mcp-session-id` header when using Streamable HTTP.
- **Persistence**: Environment variables set in one tool call (e.g., `export VAR=VAL`) are preserved and accessible in subsequent tool calls within the same MCP session.
- **Cleanup**: The `DELETE /mcp` hook is correctly triggered by the SDK when the connection is closed, allowing for immediate cleanup of host-side terminal processes.
- **Idle Reaping**: The background reaper loop provides a safety net for cleaning up sessions that weren't explicitly terminated.

## Conclusion
The `openhands-software-agent-sdk` successfully maintains session affinity in Streamable HTTP mode, enabling stateful host-side tools like persistent bash terminals. This allows agents to have a continuous, stateful interaction with the host environment throughout their session.

## Usage
To run the experiment:
1. Start the server:
   ```bash
   export MCP_PORT=8001
   uv run python experiment/mcp_server/terminal_server.py
   ```
2. Run the agent script:
   ```bash
   uv run python experiment/agent/terminal_agent_script.py
   ```

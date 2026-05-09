# Experiment: MCP Session ID Validation in Streamable HTTP

## Objective
To validate whether the `openhands-software-agent-sdk` maintains a consistent MCP session ID across multiple tool calls when using the Model Context Protocol (MCP) in **Streamable HTTP** mode.

## Methodology
The experiment uses a custom MCP server and an agent script to track session continuity.

### 1. MCP Server Setup
A `fastmcp` server was implemented that:
- Exposes a tool `get_session_id`.
- Returns the `ctx.session_id` provided by the `fastmcp` context.
- Logs the incoming HTTP headers to verify the presence of `mcp-session-id`.
- Supports both `sse` and `streamable-http` transports.

### 2. Agent Setup
An SDK-based agent was configured with the local MCP server and instructed to:
- Call the `get_session_id` tool.
- Perform a second call to the same tool within the same conversation.
- Report the results from both calls.

## Results

### Transport: SSE
In SSE mode, the session ID is maintained by the underlying `mcp` client and passed via the `session_id` query parameter in the message POST requests.

- **First Call**: `a9bc3c46-0165-4807-a7a3-856b643a6af5`
- **Second Call**: `a9bc3c46-0165-4807-a7a3-856b643a6af5`
- **Continuity**: Verified.

### Transport: Streamable HTTP
In Streamable HTTP mode, the SDK harness (via `fastmcp` and `mcp` libraries) explicitly manages the `mcp-session-id` header.

- **First Call**: `4a616396c284447ca74d22e7d382781b`
- **Second Call**: `4a616396c284447ca74d22e7d382781b`
- **Headers observed**:
  ```json
  {
    "host": "localhost:8000",
    "mcp-session-id": "4a616396c284447ca74d22e7d382781b",
    "mcp-protocol-version": "2025-11-25",
    ...
  }
  ```
- **Continuity**: Verified.

## Conclusion
The `openhands-software-agent-sdk` successfully maintains MCP session state across tool calls. This is because the `MCPClient` is initialized once per `Agent` instance and the underlying transport session is preserved for the duration of the conversation.

This validation confirms that stateful MCP tools (such as persistent terminal sessions) can be reliably implemented using the SDK's existing MCP integration.

## Experiment 02: Persistent Terminal Sessions
This experiment validated that host-side terminal processes can be correctly reused across agent tool calls by mapping the MCP session ID to a `TerminalExecutor` instance.

- **Status**: SUCCESS
- **Key Outcome**: Environment variables and shell state are preserved across sequential `conversation.run()` steps.
- **Detailed Report**: [02_terminal_persistence_report.md](./02_terminal_persistence_report.md)

## Experiment 03: Multi-Agent Session Isolation
This experiment confirmed that the MCP server can manage multiple isolated terminal sessions for different agents concurrently.

- **Status**: SUCCESS
- **Key Outcome**: Each agent has its own persistent shell; variables set by one agent are not visible to others.
- **Detailed Report**: [03_multi_agent_isolation_report.md](./03_multi_agent_isolation_report.md)

## Experiment 04: Concurrent Requests Handling
This experiment validated that the terminal server can process simultaneous requests from different agents without blocking.

- **Status**: SUCCESS
- **Key Outcome**: Two 10-second tasks were completed in ~17.5 seconds, proving parallel execution.
- **Detailed Report**: [04_concurrent_requests_report.md](./04_concurrent_requests_report.md)

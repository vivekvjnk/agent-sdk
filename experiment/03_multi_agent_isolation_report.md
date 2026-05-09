# Experiment 03: Multi-Agent Terminal Session Isolation

## Objective
To validate that the MCP terminal tool server can handle separate, isolated terminal sessions for multiple agents simultaneously using the same MCP server instance, and that each agent's session remains persistent and private.

## Methodology
1. **MCP Server**: The `terminal_server.py` from Experiment 02 was used, running on port 8801.
2. **Agents**: Two independent agents (`Agent-1` and `Agent-2`) were initialized using the same `LLM` and `MCPConfig`.
3. **Execution Steps**:
    - **Step 1**: Agent 1 exports `AGENT_1_VAR=Value-1`.
    - **Step 2**: Agent 2 exports `AGENT_2_VAR=Value-2`.
    - **Step 3**: Agent 1 verifies `AGENT_1_VAR` is still set and checks if `AGENT_2_VAR` is visible.
    - **Step 4**: Agent 2 verifies `AGENT_2_VAR` is still set and checks if `AGENT_1_VAR` is visible.

## Key Findings
- **Isolation**: Agent 1 could NOT see variables set by Agent 2, and vice versa. This confirms that the `mcp-session-id` (which is unique per `Conversation`) successfully isolates `TerminalExecutor` instances in the server's `SessionTerminalRegistry`.
- **Persistence**: Both agents maintained their own shell state across multiple turns in their respective conversations.
- **Concurrent Management**: The server correctly managed multiple `TerminalExecutor` instances in parallel, keyed by the `mcp-session-id` extracted from the Streamable HTTP headers.

## Conclusion
The implementation of the `SessionTerminalRegistry` and the use of `ctx.session_id` in `fastmcp` provide a robust foundation for multi-agent support. Each conversation with an agent gets its own dedicated, persistent, and isolated terminal environment on the host machine.

## Usage
To run the experiment:
1. Ensure the server is running:
   ```bash
   export MCP_PORT=8801
   uv run python experiment/mcp_server/terminal_server.py
   ```
2. Run the multi-agent script:
   ```bash
   uv run python experiment/agent/multi_agent_terminal_script.py
   ```

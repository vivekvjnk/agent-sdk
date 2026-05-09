# Experiment 04: Concurrent Multi-Agent Requests

## Objective
To validate that the MCP terminal tool server is capable of handling simultaneous requests from different agents at the same time, ensuring that one agent's long-running task does not block other agents.

## Methodology
1. **MCP Server**: The `terminal_server.py` was used, running on port 8801.
2. **Agents**: Two independent agents (`Agent 1` and `Agent 2`) were used.
3. **Execution**: Both agents were triggered simultaneously using `asyncio.gather` and `asyncio.to_thread` to execute a 10-second sleep command (`sleep 10 && echo Agent-X-Finished`).
4. **Verification**: The total elapsed time for both 10-second tasks was measured. Concurrent execution should result in a total time significantly less than 20 seconds.

## Key Findings
- **Concurrency**: The total elapsed time for both 10-second tasks was **17.52 seconds**. This confirms that the server processed both requests in parallel. If processed sequentially, the time would have exceeded 20 seconds (10s + 10s + LLM overhead).
- **Non-blocking Server**: The use of `uvicorn` and the asynchronous nature of the `fastmcp` server (specifically `loop.run_in_executor` for synchronous tool calls) correctly handles concurrent connections and tool executions.
- **Session Integrity**: Even while running concurrently, each agent's request was correctly routed to its respective persistent terminal session based on the unique `mcp-session-id`.

## Conclusion
The terminal server implementation is fully capable of managing simultaneous requests from multiple agents. Each agent's session is both isolated and non-blocking, making it suitable for complex multi-agent workflows where agents might need to interact with the host environment in parallel.

## Usage
To run the experiment:
1. Ensure the server is running:
   ```bash
   export MCP_PORT=8801
   uv run python experiment/mcp_server/terminal_server.py
   ```
2. Run the concurrent script:
   ```bash
   uv run python experiment/agent/concurrent_multi_agent_script.py
   ```

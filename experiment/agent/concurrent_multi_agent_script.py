import asyncio
import os
import time
from pydantic import SecretStr
from openhands.sdk import (
    LLM,
    Agent,
    AgentContext,
    Conversation,
)
from openhands.sdk.event import MessageEvent

def get_last_agent_message(conversation):
    for event in reversed(conversation.state.events):
        if isinstance(event, MessageEvent) and event.source == "agent":
            return event.llm_message.content
    return "No response from agent."

async def run_agent_task(name, agent, command):
    print(f"[{name}] Starting task: {command}")
    start_time = time.time()
    conv = Conversation(agent)
    conv.send_message(command)
    # Since conv.run() is synchronous, we run it in a thread to achieve concurrency
    await asyncio.to_thread(conv.run)
    end_time = time.time()
    response = get_last_agent_message(conv)
    print(f"[{name}] Finished in {end_time - start_time:.2f}s. Response: {response.strip()}")
    return response

async def run_experiment():
    # 0. Setup LLM
    api_key = os.getenv("LLM_API_KEY", "dummy")
    model = os.getenv("LLM_MODEL", "anthropic/claude-3-5-sonnet-20240620")
    base_url = os.getenv("LLM_BASE_URL")

    llm = LLM(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
    )

    # 1. Setup MCP Server Configuration
    mcp_config = {
        "mcpServers": {
            "terminal-session-server": {
                "url": "http://localhost:8801/mcp",
                "transport": "streamable-http"
            }
        }
    }

    # 2. Initialize Agents
    agent1 = Agent(
        agent_context=AgentContext(system_message_suffix="You are Agent 1."),
        llm=llm,
        mcp_config=mcp_config
    )
    
    agent2 = Agent(
        agent_context=AgentContext(system_message_suffix="You are Agent 2."),
        llm=llm,
        mcp_config=mcp_config
    )

    print("\n--- Starting Concurrent Multi-Agent Experiment ---")
    print("Both agents will run a 10-second sleep command simultaneously.")
    
    # We want them to actually run the bash command in parallel.
    command1 = "Run 'sleep 10 && echo Agent-1-Finished'"
    command2 = "Run 'sleep 10 && echo Agent-2-Finished'"

    start = time.time()
    
    # Run both agents concurrently
    results = await asyncio.gather(
        run_agent_task("Agent 1", agent1, command1),
        run_agent_task("Agent 2", agent2, command2)
    )
    
    total_time = time.time() - start
    print(f"\nTotal elapsed time for both 10s tasks: {total_time:.2f}s")
    
    # If they were sequential, it would be > 20s (since each takes 10s + LLM overhead)
    if total_time < 18: 
        print(f"SUCCESS: Tasks ran concurrently! Total time: {total_time:.2f}s")
    else:
        print(f"FAILURE: Tasks did not appear to run concurrently. Total time: {total_time:.2f}s")

    print("\n--- Experiment Finished ---")

if __name__ == "__main__":
    asyncio.run(run_experiment())

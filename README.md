# OpenHands Agent SDK

Build AI agents that write software. A clean, modular SDK with production-ready tools.

The OpenHands SDK allows you to build applications with agents that write software. This SDK also powers [OpenHands](https://github.com/OpenHands/OpenHands), an all-batteries-included coding agent that you can access through a GUI, CLI, or API.

## Features

- **Single Python API**: Unified interface for building coding agents with minimal boilerplate
- **Pre-defined Tools**: Built-in tools for bash commands, file editing, task tracking, and web browsing
- **REST-based Agent Server**: Deploy agents as scalable web services with WebSocket support for real-time interactions

## Why OpenHands Agent SDK?

- **Emphasis on coding**: Purpose-built for software development tasks with specialized tools and workflows
- **State-of-the-Art Performance**: Powered by advanced LLMs and optimized for real-world coding scenarios
- **Free and Open Source**: MIT licensed with an active community and transparent development

## Quick Start

Here's what building with the SDK looks like:

```python
from openhands.sdk import LLM, Conversation
from openhands.tools.preset.default import get_default_agent

# Configure LLM and create agent
llm = LLM(model="openhands/claude-sonnet-4-5-20250929", api_key=api_key)
agent = get_default_agent(llm=llm)

# Start a conversation
conversation = Conversation(agent=agent, workspace="/path/to/project")
conversation.send_message("Write 3 facts about this project into FACTS.txt.")
conversation.run()
```

For installation instructions and detailed setup, see the [Getting Started Guide](https://docs.openhands.dev/sdk/getting-started).

## Documentation

For detailed documentation, tutorials, and API reference, visit:

**[https://docs.openhands.dev/sdk](https://docs.openhands.dev/sdk)**

The documentation includes:
- [Getting Started Guide](https://docs.openhands.dev/sdk/getting-started) - Installation and setup
- [Architecture & Core Concepts](https://docs.openhands.dev/sdk/arch/overview) - Agents, tools, workspaces, and more
- [Guides](https://docs.openhands.dev/sdk/guides/hello-world) - Hello World, custom tools, MCP, skills, and more
- [API Reference](https://docs.openhands.dev/sdk/guides/agent-server/api-reference/server-details/alive) - Agent Server REST API documentation

## Examples

The `examples/` directory contains comprehensive usage examples:

- **Standalone SDK** (`examples/01_standalone_sdk/`) - Basic agent usage, custom tools, and microagents
- **Remote Agent Server** (`examples/02_remote_agent_server/`) - Client-server architecture and WebSocket connections
- **GitHub Workflows** (`examples/03_github_workflows/`) - CI/CD integration and automated workflows

## Contributing

For development setup, testing, and contribution guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Community

- [Join Slack](https://openhands.dev/joinslack) - Connect with the OpenHands community
- [GitHub Repository](https://github.com/OpenHands/agent-sdk) - Source code and issues
- [Documentation](https://docs.openhands.dev/sdk) - Complete documentation

## License

<<<<<<< HEAD
MIT License - see [LICENSE](LICENSE) for details.
=======
## Core Concepts

### Agents

Agents are the central orchestrators that coordinate between LLMs and tools. The SDK provides two main approaches for creating agents:

#### Using Default Presets

We recommend that you try out the default presets at first, which gives you a powerful agent with our default set of tools.

```python
from openhands.tools.preset import get_default_agent

# Get a fully configured agent with default tools and settings
agent = get_default_agent(
    llm=llm,
    cli_mode=True,  # Disable browser tools for CLI environments
)
```

#### Manual Agent Configuration

```python
from openhands.sdk import Agent
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.execute_bash import BashTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool

# Register tools
register_tool("BashTool", BashTool)
register_tool("FileEditorTool", FileEditorTool)
register_tool("TaskTrackerTool", TaskTrackerTool)

# Create agent with explicit tool specifications
agent = Agent(
    llm=llm,
    tools=[
        Tool(name="BashTool", params={"working_dir": os.getcwd()}),
        Tool(name="FileEditorTool"),
        Tool(name="TaskTrackerTool", params={"save_dir": os.getcwd()}),
    ],
)
```

### LLM Integration

The SDK supports multiple LLM providers through a unified interface:

```python
from openhands.sdk import LLM, LLMRegistry
from pydantic import SecretStr

# Direct LLM configuration
llm = LLM(
    model="gpt-4",
    api_key=SecretStr("your-api-key"),
    base_url="https://api.openai.com/v1"
)

# Using LLM registry for shared configurations
registry = LLMRegistry()
registry.add("default", llm)
llm = registry.get("default")
```

#### Steps to setup VertexAI 
1. Set environment variables
- VERTEXAI_PROJECT = <project id>
- VERTEXAI_LOCATION = <project location>
- GOOGLE_APPLICATION_CREDENTIALS = <path to google cloud json credential file>
- LLM_MODEL="vertex_ai/gemini-2.5-flash" #replace "gemini-2.5-flash with model of choice 
- LLM_API_KEY="abc" # Set this variable to any value, to avoid assert errors

NOTES:
- These environment variables should be set in the host system from where `uv run ...` is called
- If code is running on a container, make sure these environment variables are set properly in the container

### Tools

Tools provide agents with capabilities to interact with the environment. The SDK includes several built-in tools:

- **BashTool**: Execute bash commands in a persistent shell session
- **FileEditorTool**: Create, edit, and manage files with advanced editing capabilities  
- **TaskTrackerTool**: Organize and track development tasks systematically
- **BrowserToolSet**: Automate web browser interactions (disabled in CLI mode)

The default contains all of these tools, but for more control, you can configure tools explicitly:

```python
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.execute_bash import BashTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool

# Register tools
register_tool("BashTool", BashTool)
register_tool("FileEditorTool", FileEditorTool)
register_tool("TaskTrackerTool", TaskTrackerTool)

# Create tool specifications
tools = [
    Tool(name="BashTool", params={"working_dir": os.getcwd()}),
    Tool(name="FileEditorTool"),
    Tool(name="TaskTrackerTool", params={"save_dir": os.getcwd()}),
]
```

### Conversations

Conversations manage the interaction flow between users and agents:

```python
from openhands.sdk import Conversation

conversation = Conversation(agent=agent)

# Send messages
conversation.send_message("Your request here")

# Execute the conversation until the agent enters "await user input" state
conversation.run()
```

### Context Management

The context system manages agent state, environment, and conversation history.

Context is automatically managed but you can customize your context with:

1. [Repo Microagents](https://docs.all-hands.dev/usage/prompting/microagents-repo) that provide agent with context of your repository.
2. [Knowledge Microagents](https://docs.all-hands.dev/usage/prompting/microagents-keyword) that provide agent with context when user mentioned certain keywords
3. Providing custom suffix for system and user prompt.

```python
from openhands.sdk import AgentContext
from openhands.sdk.context import RepoMicroagent, KnowledgeMicroagent

context = AgentContext(
    microagents=[
        RepoMicroagent(
            name="repo.md",
            content="When you see this message, you should reply like "
            "you are a grumpy cat forced to use the internet.",
        ),
        KnowledgeMicroagent(
            name="flarglebargle",
            content=(
                'IMPORTANT! The user has said the magic word "flarglebargle". '
                "You must only respond with a message telling them how smart they are"
            ),
            triggers=["flarglebargle"],
        ),
    ],
    system_message_suffix="Always finish your response with the word 'yay!'",
    user_message_suffix="The first character of your response should be 'I'",
)
```

## Agent Server

The SDK includes a REST API and WebSocket server for remote agent interactions:

```python
from openhands.agent_server.api import create_app
import uvicorn

# Create FastAPI application
app = create_app()

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

The agent server provides:
- REST API endpoints for agent management
- WebSocket connections for real-time conversations
- Authentication and session management
- Scalable deployment options

### API Endpoints

- `POST /conversations` - Create new conversation
- `GET /conversations/{id}` - Get conversation details
- `POST /conversations/{id}/messages` - Send message to conversation
- `WebSocket /ws/{conversation_id}` - Real-time conversation updates

## Development Workflow

### Environment Setup

```bash
# Initial setup
make build

# Install additional dependencies
# add `--dev` if you want to install 
uv add package-name

# Update dependencies
uv sync
```

### Code Quality

The project enforces strict code quality standards:

```bash
# Format code
make format

# Lint code
make lint

# Run pre-commit hooks
uv run pre-commit run --all-files

# Type checking (included in pre-commit)
uv run pyright
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test suite
uv run pytest tests/cross/
uv run pytest tests/sdk/
uv run pytest tests/tools/

# Run with coverage
uv run pytest --cov=openhands --cov-report=html
```

### Pre-commit Workflow

Before every commit:

```bash
# Run on specific files
uv run pre-commit run --files path/to/file.py

# Run on all files
uv run pre-commit run --all-files
```

>>>>>>> 38fa76d (Added steps to configure VertexAI LLM models)

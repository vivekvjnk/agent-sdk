from pathlib import Path
import os

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    ImageContent,
    LLMConvertibleEvent,
    Message,
    TextContent,
    get_logger,
)
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from prompts import VISION_PROMPT

logger = get_logger(__name__)


submodule_root = Path(__file__).resolve().parent

def run_vision_agent(image_path: str, question: str) -> str:
    # Configure LLM (use a vision-capable model)
    api_key = os.getenv("LLM_API_KEY")
    assert api_key is not None, "LLM_API_KEY environment variable is not set."
    model = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
    base_url = os.getenv("LLM_BASE_URL")
    llm = LLM(
        usage_id="vision-agent",
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        temperature=1.0,
        reasoning_effort="low"
    )
    assert llm.vision_is_active(), "The selected LLM model does not support vision input."

    cwd = os.getcwd()

    system_prompt_file_path = str(Path(submodule_root/"vision_agent_prompt.j2"))
    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
        ],
        system_prompt_filename=system_prompt_file_path,
    )

    llm_messages = []
    
    def conversation_callback(event: Event) -> None:
        if isinstance(event, LLMConvertibleEvent):
            llm_messages.append(event.to_llm_message())

    conversation = Conversation(
        agent=agent,
        workspace=cwd,
        callbacks=[conversation_callback],
    )

    import base64
    abs_image_path = os.path.abspath(image_path)
    with open(abs_image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(abs_image_path)[1].lower().replace(".", "")
    if ext == "jpg": ext = "jpeg"
    file_uri = f"data:image/{ext};base64,{img_b64}"
    
    conversation.send_message(
        Message(
            role="user",
            content=[
                TextContent(text=f"Please analyze this image.\nPath: {abs_image_path}\nQuestion: {question}"),
                ImageContent(image_urls=[file_uri]),
            ],
        )
    )

    runner = conversation.run()
    
    # Returning the final message as string as a simplified format
    last_msg = ""
    if llm_messages:
        # get the last assistant message
        for msg in reversed(llm_messages):
            if hasattr(msg, "role") and msg.role == "assistant":
                last_msg = str(msg.content)
                break
    return last_msg


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m examples.06_custom_examples.vision_agent.vision_agent <image_path> \"<question>\"")
        sys.exit(1)
    
    img_path = sys.argv[1]
    quest = sys.argv[2]
    image_path = Path(submodule_root / img_path)
    answer = run_vision_agent(img_path, quest)
    print("Agent Final Answer:")
    print(answer)

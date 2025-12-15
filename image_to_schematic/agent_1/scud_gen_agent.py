import os
from pathlib import Path
from typing import List

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Message,
    TextContent,
    ImageContent,
    get_logger,
)
from openhands.sdk.tool.spec import Tool
from openhands.tools.file_editor import FileEditorTool
from image_to_schematic.agent_1.prompts import build_agent_1_system_prompt
from pathlib import Path
from typing import List

from google.cloud import storage
from openhands.sdk import get_logger

logger = get_logger(__name__)


def run_agent_1_scud_builder(
    schematic_image_urls: List[str],
    workspace_root: Path,
    model: str = "google/gemini-1.5-pro-002",
):
    """
    Agent 1: Incrementally builds the SCUD (Shared Circuit Understanding Document)
    from schematic image crops.
    """

    workspace_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using Agent 1 workspace: {workspace_root}")

    llm = LLM(
        usage_id="agent-1-llm",
        model=model,
        api_key=SecretStr(os.getenv("LLM_API_KEY")),
    )

    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=FileEditorTool.name),
        ],
    )

    conversation = Conversation(
        agent=agent,
        workspace=str(workspace_root),
    )

    # ------------------------------------------------------------------
    # Send system persona
    # ------------------------------------------------------------------
    conversation.send_message(
        Message(
            role="user",
            content=[TextContent(text=build_agent_1_system_prompt())],
        )
    )

    # ------------------------------------------------------------------
    # Feed schematic images incrementally
    # ------------------------------------------------------------------
    for idx, image_url in enumerate(schematic_image_urls):
        if idx == 0:
            instruction = """
This is the first schematic image.

If `scud.md` does not exist:
- Create it using the mandatory four-section structure.

Then interpret the image and populate the document
based only on what is visible or reasonably implied.
""".strip()
        else:
            instruction = """
A new schematic crop is provided.

- Read the existing `scud.md` fully.
- Update it incrementally based on new information.
- Do NOT rewrite the document from scratch.
- If this image contradicts earlier understanding,
  record the contradiction explicitly.
""".strip()

        conversation.send_message(
            Message(
                role="user",
                content=[
                    TextContent(text=instruction),
                    ImageContent(image_urls=[image_url]),
                ],
            )
        )

        conversation.run()

        logger.info(f"Processed schematic image {idx + 1}/{len(schematic_image_urls)}")

    logger.info("Agent 1 completed SCUD construction")
    logger.info(f"Total cost: {llm.metrics.accumulated_cost}")




def run_scud_builder_pipeline(
    gcs_bucket_name: str,
    workspace_root: Path,
    images_prefix: str = "schematic_subcircuits/images",
):
    """
    Top-level orchestration for Agent 1 (SCUD Builder).

    Responsibilities:
    - List schematic image blobs from GCS
    - Resolve them into gs:// URLs
    - Order them deterministically
    - Feed them incrementally to Agent 1

    Assumptions:
    - All schematic images are already uploaded to GCS
    - images_prefix contains only schematic-related images
    """

    # ------------------------------------------------------------------
    # GCS Setup (read-only)
    # ------------------------------------------------------------------
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)

    images_prefix = images_prefix.rstrip("/") + "/"

    logger.info(
        f"Listing schematic images under "
        f"gs://{gcs_bucket_name}/{images_prefix}"
    )

    # ------------------------------------------------------------------
    # List and filter image blobs
    # ------------------------------------------------------------------
    image_blobs = [
        blob
        for blob in storage_client.list_blobs(bucket, prefix=images_prefix)
        if blob.name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_blobs:
        raise RuntimeError(
            f"No image blobs found under "
            f"gs://{gcs_bucket_name}/{images_prefix}"
        )

    # Deterministic ordering (important for reproducibility)
    image_blobs.sort(key=lambda b: b.name)

    schematic_image_urls: List[str] = [
        f"gs://{gcs_bucket_name}/{blob.name}"
        for blob in image_blobs
    ]

    logger.info(
        f"Found {len(schematic_image_urls)} schematic images for SCUD builder"
    )

    for url in schematic_image_urls:
        logger.info(f"  → {url}")

    # ------------------------------------------------------------------
    # Run Agent 1 (SCUD Builder)
    # ------------------------------------------------------------------
    
    model = os.getenv("LLM_MODEL", "vertex_ai/gemini-2.5-flash")
    run_agent_1_scud_builder(
        schematic_image_urls=schematic_image_urls,
        workspace_root=workspace_root,
        model=model,
    )

    logger.info("SCUD builder pipeline completed successfully")

if __name__ == "__main__":
    run_scud_builder_pipeline(
        gcs_bucket_name="vhl",
        workspace_root=Path("./image_to_schematic/agent_1/workspace"),
    )

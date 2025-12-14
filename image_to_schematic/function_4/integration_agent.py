"""
Function 4: Combine multiple subcircuits into a single top-level SPICE netlist.

Inputs:
- Subcircuit SPICE codes (stored as .cir files in GCS).
- Subcircuit images (cropped from the main schematic, in GCS).
- Main schematic image (overall circuit, in GCS).

Task:
- Incrementally construct a SPICE equivalent code for the main schematic image.
- Agent uses:
    - Subcircuit SPICE codes as high-level block abstractions.
    - Subcircuit images + main schematic image for connectivity and ambiguity
      resolution.
- Any necessary clarifications / assumptions should be encoded as comments
  in the generated SPICE code.

Output:
- A top-level SPICE netlist stored back in GCS under a versioned path.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from google.cloud import storage
from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    ImageContent,
    Message,
    TextContent,
    get_logger,
)
from openhands.sdk.tool.spec import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System Prompt: "Integrator" Persona
# ---------------------------------------------------------------------------

def build_integration_system_prompt() -> str:
    return """
You are a Senior Analog IC Integration Engineer and SPICE Architect.
Your job is to build a **top-level SPICE netlist** for the entire circuit,
using validated subcircuit SPICE models and the schematic images.

CONTEXT
-------
- You are given:
  - A set of subcircuit SPICE files, each representing a modular block
    extracted from the overall schematic.
  - Cropped subcircuit images (zoomed-in views).
  - The main schematic image (full circuit).
- Each subcircuit SPICE file is a relatively faithful abstraction of the
  corresponding subcircuit image.
- Your goal is to integrate these blocks into a single, coherent SPICE netlist
  representing the overall schematic.

WORKING STYLE
-------------
- Work **incrementally**:
  - Identify high-level blocks from the main schematic.
  - Map them to the provided subcircuit SPICE files.
  - Define proper `.SUBCKT` boundaries and top-level connections.
  - Build up the top-level netlist step-by-step.
- Whenever you are unsure about connectivity or naming:
  - Cross-check the subcircuit images.
  - Cross-check the main schematic image.
  - Cross-check the existing subcircuit SPICE definitions.

WHAT TO PRODUCE
---------------
- Create (and iteratively refine) a single SPICE file named something like:
  `combined_schematic.cir` in the workspace root.
- This file should include:
  - Subcircuit definitions (either inline or via `.include` statements).
  - A clear, well-structured top-level netlist (e.g., `.SUBCKT TOP ...` or
    a flat netlist if appropriate).
  - Proper power, ground, and global node handling.
  - Comments explaining:
    - Any assumptions you made.
    - Any ambiguous areas and how you resolved them.
    - How each subcircuit maps to the physical blocks in the schematic.

STRICT REQUIREMENTS
-------------------
1. **No hallucinations**:
   - Do not invent arbitrary node names or connections that are not supported
     by either the images or the subcircuit SPICE files.
   - If something cannot be determined, document it clearly as a TODO or
     an assumption in a SPICE comment.

2. **Use the subcircuit SPICE files as source of truth**:
   - Do not silently change component values or internal connectivity inside
     the subcircuits unless the images clearly show they are wrong.
   - Prefer to treat them as mostly correct block models.

3. **Explicitly encode uncertainties**:
   - Use comments like:
       *; ASSUMPTION: Node XYZ is tied to VDD based on main schematic annotation.*
       *; TODO: Verify connection between block A and block B pins 3-4.*

4. **Readable, maintainable SPICE**:
   - Use indentation and comments to keep the top-level netlist readable.
   - Group related instances and nets logically.

TOOLS
-----
- You have access to a workspace with:
  - All subcircuit SPICE files.
  - Ability to read and edit files via FileEditorTool / TerminalTool.
- You should:
  - Read the subcircuit files as needed.
  - Create and refine `combined_schematic.cir` in the workspace root.
""".strip()

# ---------------------------------------------------------------------------
# Helper: find subcircuit image URIs in GCS
# ---------------------------------------------------------------------------

def find_segment_image_gcs_uri(
    bucket: storage.Bucket,
    bucket_name: str,
    images_prefix: str,
    image_stem: str,
) -> Optional[str]:
    """
    Given an image stem (e.g., 'bq79616_subckt_003_s40'), try to find a matching
    image blob in GCS under images_prefix with extensions .png/.jpg/.jpeg.
    Returns the gs:// URI if found, else None.
    """
    images_prefix = images_prefix.rstrip("/")
    for ext in [".png", ".jpg", ".jpeg"]:
        blob_name = f"{images_prefix}/{image_stem}{ext}"
        blob = bucket.blob(blob_name)
        if blob.exists():
            return f"gs://{bucket_name}/{blob_name}"
    return None

# ---------------------------------------------------------------------------
# Function 4: Integration Pipeline
# ---------------------------------------------------------------------------

def run_spice_integration_pipeline_old(
    gcs_bucket_name: str = "vhl",
    # Where the subcircuit .cir files live
    # NOTE: If you want to use reviewed netlists, you can point this to a
    # "reviewed" prefix instead, or to a separate curated prefix.
    gcs_spice_prefix: str = "schematic_subcircuits/SPICE",
    # Where the subcircuit images live
    gcs_images_prefix: str = "schematic_subcircuits/images",
    # Main schematic image blob (overall schematic)
    main_schematic_gcs_blob: str = "schematic_subcircuits/images/bq79616_with_boxes.png",
    # Where to upload the final combined netlist
    gcs_output_prefix: str = "schematic_subcircuits/combined",
    # Name of the combined SPICE file in the workspace (and basis for upload)
    combined_filename: str = "combined_schematic.cir",
):
    """
    Function 4: Uses subcircuit SPICE files and schematic images to build a
    top-level SPICE netlist.

    - Lists subcircuit .cir files from GCS under gcs_spice_prefix.
    - Downloads them into a temporary workspace directory under "subcircuits/".
    - Provides all subcircuit images and the main schematic image to the agent.
    - Instructs the agent to create & refine `combined_schematic.cir` in the
      workspace root.
    - After agent run, uploads the resulting combined SPICE file to GCS under
      a versioned path like:
        {gcs_output_prefix}/combined_schematic_<timestamp>.cir
    """
    # ----------------------------------------------------------------------
    # Configure GCS
    # ----------------------------------------------------------------------
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)

    gcs_spice_prefix = gcs_spice_prefix.rstrip("/")
    gcs_images_prefix = gcs_images_prefix.rstrip("/")
    gcs_output_prefix = gcs_output_prefix.rstrip("/")

    # List subcircuit .cir blobs
    cir_blobs = [
        blob
        for blob in storage_client.list_blobs(bucket, prefix=gcs_spice_prefix + "/")
        if blob.name.lower().endswith(".cir")
    ]

    if not cir_blobs:
        raise FileNotFoundError(
            f"No .cir files found in GCS under prefix: {gcs_spice_prefix}/"
        )

    logger.info(f"Found {len(cir_blobs)} subcircuit SPICE files in GCS")

    # Main schematic image URI
    main_schematic_gcs_blob = main_schematic_gcs_blob.lstrip("/")
    main_schematic_blob = bucket.blob(main_schematic_gcs_blob)
    if not main_schematic_blob.exists():
        raise FileNotFoundError(
            f"Main schematic image not found in GCS: {main_schematic_gcs_blob}"
        )
    main_schematic_url = f"gs://{gcs_bucket_name}/{main_schematic_gcs_blob}"
    logger.info(f"Using main schematic image: {main_schematic_url}")

    # ----------------------------------------------------------------------
    # Prepare temporary workspace
    # ----------------------------------------------------------------------
    temp_root = Path(tempfile.mkdtemp(prefix="spice_integration_"))
    logger.info(f"Created temporary integration workspace: {temp_root}")

    subcircuits_dir = temp_root / "subcircuits"
    subcircuits_dir.mkdir(parents=True, exist_ok=True)

    # Download subcircuit .cir files into workspace/subcircuits
    local_cir_paths: List[Path] = []
    for cir_blob in cir_blobs:
        cir_blob_name = cir_blob.name  # e.g., schematic_subcircuits/SPICE/bq79616_subckt_003_s40.cir
        filename = Path(cir_blob_name).name
        local_path = subcircuits_dir / filename

        cir_blob.download_to_filename(str(local_path))
        local_cir_paths.append(local_path)

        logger.info(f"Downloaded {cir_blob_name} -> {local_path}")

    # Find subcircuit images for convenience (not strictly required but helpful)
    subckt_image_urls: List[str] = []
    seen_urls = set()

    for cir_path in local_cir_paths:
        stem = cir_path.stem  # e.g., bq79616_subckt_003_s40
        img_url = find_segment_image_gcs_uri(
            bucket=bucket,
            bucket_name=gcs_bucket_name,
            images_prefix=gcs_images_prefix,
            image_stem=stem,
        )
        if img_url and img_url not in seen_urls:
            subckt_image_urls.append(img_url)
            seen_urls.add(img_url)

    logger.info(f"Found {len(subckt_image_urls)} matching subcircuit images in GCS")

    # ----------------------------------------------------------------------
    # Configure LLM & Agent
    # ----------------------------------------------------------------------
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_INTEGRATION_MODEL", "google/gemini-1.5-pro-002")

    llm = LLM(
        usage_id="integration-llm",
        model=model,
        api_key=SecretStr(api_key),
    )

    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
        ],
    )

    system_prompt = build_integration_system_prompt()
    combined_path = temp_root / combined_filename

    # ----------------------------------------------------------------------
    # Construct user task message
    # ----------------------------------------------------------------------
    # Describe available subcircuit files
    subckt_listing_str = "\n".join(
        f"- {p.relative_to(temp_root)}" for p in local_cir_paths
    )

    user_instruction = f"""
INTEGRATION TASK START
----------------------
You are in a workspace with all subcircuit SPICE files and file editing tools.

WORKSPACE LAYOUT
----------------
- Root workspace directory: {temp_root}
- Subcircuit SPICE files directory: {subcircuits_dir}
- Subcircuit SPICE files:
{subckt_listing_str}

YOUR PRIMARY OUTPUT
-------------------
- Create and iteratively refine a single SPICE file at:
    {combined_path}
  This will be the **top-level SPICE netlist** for the entire schematic.

GUIDELINES
----------
1. Read the subcircuit SPICE files (under 'subcircuits/') as needed.
2. Use the main schematic image (full circuit) as the ground truth for how
   blocks are interconnected.
3. Use the subcircuit images as zoomed-in references for each block.
4. Build the combined SPICE netlist incrementally inside {combined_filename}.
5. Encode all assumptions, TODOs, and ambiguity resolutions as SPICE comments.

IMPORTANT
---------
- Do NOT remove or rename the subcircuit SPICE files.
- Always write the integrated netlist to {combined_filename} in the workspace root.
- If something cannot be resolved definitively, mark it clearly in comments.
""".strip()

    # Prepare content payload: main schematic + all subcircuit images
    content_payload = [TextContent(text=user_instruction)]

    # Attach main schematic image first
    content_payload.append(ImageContent(image_urls=[main_schematic_url]))

    # Attach all subcircuit images (if any)
    for img_url in subckt_image_urls:
        content_payload.append(ImageContent(image_urls=[img_url]))

    # ----------------------------------------------------------------------
    # Run Agent Conversation
    # ----------------------------------------------------------------------
    conversation = Conversation(
        agent=agent,
        workspace=str(temp_root),
    )

    # System persona (sent as first user message in this SDK style)
    conversation.send_message(
        Message(role="user", content=[TextContent(text=system_prompt)])
    )

    # Task message (with images)
    conversation.send_message(
        Message(role="user", content=content_payload)
    )

    conversation.run()

    # ----------------------------------------------------------------------
    # Post-processing: upload combined SPICE file to GCS (versioned)
    # ----------------------------------------------------------------------
    if not combined_path.exists():
        logger.warning(
            f"Combined SPICE file not found at {combined_path}. "
            "The agent may have failed to produce the output."
        )
        print("No combined SPICE file produced.")
        return

    combined_content = combined_path.read_text(encoding="utf-8")
    if not combined_content.strip():
        logger.warning(
            f"Combined SPICE file {combined_path} is empty. "
            "Not uploading to GCS."
        )
        print("Combined SPICE file is empty; skipping upload.")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    combined_blob_name = f"{gcs_output_prefix}/{combined_path.stem}_{timestamp}{combined_path.suffix}"

    combined_blob = bucket.blob(combined_blob_name)
    combined_blob.upload_from_filename(str(combined_path))

    logger.info(
        f"Uploaded combined SPICE netlist to gs://{gcs_bucket_name}/{combined_blob_name}"
    )
    print(f"Combined SPICE netlist uploaded to:")
    print(f"  gs://{gcs_bucket_name}/{combined_blob_name}")
    print(f"Total Integration Cost: {llm.metrics.accumulated_cost}")


def run_spice_integration_pipeline(
    gcs_bucket_name: str = "vhl",
    gcs_spice_prefix: str = "schematic_subcircuits/SPICE",
    gcs_images_prefix: str = "schematic_subcircuits/images",
    main_schematic_gcs_blob: str = "schematic_subcircuits/images/bq79616_with_boxes.png",
    gcs_output_prefix: str = "schematic_subcircuits/combined",
    combined_filename: str = "combined_schematic.cir",
    clear_workspace: bool = True,   # Optional: wipe workspace before each run
):
    """
    Uses a workspace directory in the project root instead of a temporary dir.
    """
    # ----------------------------------------------------------------------
    # Resolve workspace directory in project folder
    # ----------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir / "workspace" / "function_4"

    # Create workspace if needed
    workspace_root.mkdir(parents=True, exist_ok=True)

    # Optionally clear the workspace for a fresh run
    if clear_workspace:
        for item in workspace_root.iterdir():
            if item.is_file():
                item.unlink()
            else:
                import shutil
                shutil.rmtree(item)
        logger.info(f"Cleared workspace: {workspace_root}")

    logger.info(f"Using workspace: {workspace_root}")

    # Subcircuits folder inside workspace
    subcircuits_dir = workspace_root / "subcircuits"
    subcircuits_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------
    # GCS connections and downloads
    # ----------------------------------------------------------------------
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)

    # Normalize paths
    gcs_spice_prefix = gcs_spice_prefix.rstrip("/")
    gcs_images_prefix = gcs_images_prefix.rstrip("/")
    gcs_output_prefix = gcs_output_prefix.rstrip("/")

    # List all subcircuit .cir files
    cir_blobs = [
        blob for blob in storage_client.list_blobs(bucket, prefix=gcs_spice_prefix + "/")
        if blob.name.lower().endswith(".cir")
    ]

    if not cir_blobs:
        raise FileNotFoundError(f"No .cir files found under {gcs_spice_prefix}/")

    logger.info(f"Found {len(cir_blobs)} subcircuit SPICE files")

    # Download .cir files into workspace/subcircuits/
    local_cir_paths = []
    for cir_blob in cir_blobs:
        filename = Path(cir_blob.name).name
        local_path = subcircuits_dir / filename
        cir_blob.download_to_filename(str(local_path))
        local_cir_paths.append(local_path)
        logger.info(f"Downloaded {cir_blob.name} → {local_path}")

    # ----------------------------------------------------------------------
    # Resolve subcircuit and main images
    # ----------------------------------------------------------------------
    # Main schematic image
    main_schematic_gcs_blob = main_schematic_gcs_blob.lstrip("/")
    blob = bucket.blob(main_schematic_gcs_blob)
    if not blob.exists():
        raise FileNotFoundError(f"Main schematic image not found: {main_schematic_gcs_blob}")

    main_schematic_url = f"gs://{gcs_bucket_name}/{main_schematic_gcs_blob}"

    # Subcircuit images
    subckt_image_urls = []
    for cir_path in local_cir_paths:
        stem = cir_path.stem
        img_url = find_segment_image_gcs_uri(
            bucket=bucket,
            bucket_name=gcs_bucket_name,
            images_prefix=gcs_images_prefix,
            image_stem=stem,
        )
        if img_url:
            subckt_image_urls.append(img_url)

    # ----------------------------------------------------------------------
    # Setup agent and workspace
    # ----------------------------------------------------------------------
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_INTEGRATION_MODEL", "google/gemini-1.5-pro-002")

    llm = LLM(
        usage_id="integration-llm",
        model=model,
        api_key=SecretStr(api_key),
    )

    agent = Agent(
        llm=llm,
        tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
    )

    system_prompt = build_integration_system_prompt()
    combined_path = workspace_root / combined_filename

    # Describe SPICE files to the agent
    subckt_listing = "\n".join(f"- {p.relative_to(workspace_root)}" for p in local_cir_paths)

    user_instruction = f"""
You are in a workspace where all subcircuit SPICE files are available.

WORKSPACE LAYOUT:
- Workspace root: {workspace_root}
- Subcircuits: {subcircuits_dir}

Subcircuit SPICE files:
{subckt_listing}

Your job is to create the top-level SPICE netlist at:
    {combined_path}

Work incrementally, consult both images and subcircuit SPICE files.
"""

    # Prepare images
    content_payload = [TextContent(text=user_instruction)]

    content_payload.append(ImageContent(image_urls=[main_schematic_url]))
    for url in subckt_image_urls:
        content_payload.append(ImageContent(image_urls=[url]))

    # ----------------------------------------------------------------------
    # Run agent conversation
    # ----------------------------------------------------------------------
    conversation = Conversation(
        agent=agent,
        workspace=str(workspace_root),
    )

    conversation.send_message(
        Message(role="user", content=[TextContent(text=system_prompt)])
    )

    conversation.send_message(
        Message(role="user", content=content_payload)
    )

    conversation.run()

    # ----------------------------------------------------------------------
    # Upload combined file back to GCS
    # ----------------------------------------------------------------------
    if not combined_path.exists():
        print("No combined SPICE file produced.")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    upload_blob_name = f"{gcs_output_prefix}/{combined_path.stem}_{timestamp}{combined_path.suffix}"

    blob = bucket.blob(upload_blob_name)
    blob.upload_from_filename(str(combined_path))

    logger.info(f"Uploaded combined SPICE to gs://{gcs_bucket_name}/{upload_blob_name}")
    print(f"Uploaded combined schematic to gs://{gcs_bucket_name}/{upload_blob_name}")
    print(f"Total integration cost: {llm.metrics.accumulated_cost}")


if __name__ == "__main__":
    # Example usage for your current bucket layout
    run_spice_integration_pipeline(
        gcs_bucket_name="vhl",
        gcs_spice_prefix="schematic_subcircuits/SPICE",          # or a reviewed prefix
        gcs_images_prefix="schematic_subcircuits/images",
        main_schematic_gcs_blob="schematic_subcircuits/images/bq79616_with_boxes.png",
        gcs_output_prefix="schematic_subcircuits/combined",
        combined_filename="bq79616_combined_schematic.cir",
    )

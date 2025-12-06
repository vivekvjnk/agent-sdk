"""
Function 2: Schematic subcircuit → SPICE netlist using OpenHands Agent SDK.

Assumptions:
- All segmented subcircuit images are stored in a directory under the same folder
  as this script (default: ./schematic_subcircuits).
- This script will instruct the agent to write SPICE subcircuit files into another
  directory (default: ./spice_subcircuits) under the same folder.
- The LLM must be vision-capable (e.g. Claude with vision) and configured via:
    export LLM_API_KEY=...
    export LLM_MODEL=...           # optional, has a default
    export LLM_BASE_URL=...        # optional
"""

import os
from pathlib import Path
from typing import List

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
from openhands.sdk.tool.spec import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: build the system prompt for the agent
# ---------------------------------------------------------------------------

def build_system_prompt_old(spice_output_dir: str) -> str:
    """
    Create the system prompt describing the overall pipeline and the agent's role.
    """
    return f"""
You are an expert electrical engineer and SPICE modeling specialist.
You are part of a multi-step pipeline that converts schematic images into an
executable SPICE netlist.

PIPELINE OVERVIEW
-----------------
1. Function 1 (already completed) segments a large schematic into smaller
   sub-circuit images. Each image you receive is one of these subcircuits.
2. Function 2 (your role) converts EACH subcircuit image into an accurate SPICE
   subcircuit description.
3. Downstream agents may later refine or replace behavioral models, but they rely
   on your SPICE interface and connectivity being completely accurate.

YOUR ROLE
---------
Given ONE schematic subcircuit image:
- Understand the electrical structure and connectivity.
- Identify components (resistors, capacitors, diodes, BJTs, MOSFETs, passives,
  connectors, simple sources, etc.).
- Identify integrated circuits (ICs) and complex multi-pin devices.
- Produce a SPICE subcircuit (.SUBCKT) that captures ALL external behavior
  interfaces (pins and nets) and basic internal connections (for discrete parts).

IC HANDLING (VERY IMPORTANT)
----------------------------
If the schematic contains any IC or complex multi-pin device:
- DO NOT attempt to model its internal behavior.
- Instead, create a **stub SPICE subcircuit** for each IC.
- The stub MUST strictly maintain the correct pin interface:
  - Use a .SUBCKT declaration with all pins in a clear, consistent order.
  - The pin list and count must reflect the schematic as accurately as possible.
  - Example pattern:

        * IC U1: some controller
        .SUBCKT U1_stub VCC GND IN OUT FB NC1 NC2
        * Internal behavior intentionally omitted; to be filled by another agent.
        .ENDS U1_stub

- In the main subcircuit, instantiate the stub using the same pin order and
  clear net naming.

SPICE SUBCIRCUIT REQUIREMENTS
------------------------------
- Wrap the entire subcircuit logic in a .SUBCKT / .ENDS pair.
- Use a meaningful subcircuit name derived from the image or reference designator,
  e.g. SUBCKT SUBCKT_<IMAGE_NAME> or similar.
- Include all external terminals (nets that connect to other parts of the global
  schematic) as subcircuit ports.
- For discrete components:
  - Use standard SPICE primitives: R, C, L, D, Q, M, V, I, etc.
  - Use reasonable generic models if needed, but keep things simple.
- Use comments liberally to explain assumptions, unclear symbols, or inferred values.
- If any value is unreadable or ambiguous, make a clear, commented assumption:

      * R5 value unreadable; assumed 10k
      R5 NET_A NET_B 10k

NET NAMING & CONNECTIVITY
-------------------------
- Preserve any visible net labels from the schematic when possible (e.g. VCC,
  GND, OUT, IN+, IN-, VS, VBAT).
- Clearly differentiate between different ground symbols if necessary (e.g. GND,
  AGND, PGND).
- Ensure that nodes sharing the same visible label in the schematic are tied
  together in SPICE.

ITERATIVE REFINEMENT
--------------------
Within a single task for a given image, follow an internal iterative process:
1. Draft an initial SPICE subcircuit.
2. Self-review for:
   - Missing components.
   - Incorrect or missing pins for ICs.
   - Floating nodes that should not be floating.
   - Obvious syntax errors.
3. Refine and correct until the SPICE is internally consistent and ready for use.
4. Only then write the final SPICE subcircuit to the requested output file.

OUTPUT & FILE SAVING
--------------------
- The workspace root is the directory where this Python script is executed.
- Save the final SPICE subcircuit for each image into the directory:
      {spice_output_dir}
  relative to the workspace.
- Use a file name based on the image stem (e.g. image 'subckt_003.png' ->
  '{spice_output_dir}/subckt_003.cir').
- The content of each .cir file should be ONLY the SPICE subcircuit and comments.
- In your chat messages, keep prose minimal. The main artifact is the .cir file.

GENERAL STYLE
-------------
- Be precise, concise, and engineering-focused.
- Prefer correctness and clarity over fancy formatting.
- If something is genuinely ambiguous, call it out in comments near the
  corresponding SPICE elements.
    """.strip()

def build_system_prompt(spice_output_dir: str) -> str:
    """
    Create the system prompt describing the overall pipeline and the agent's role.
    """
    return f"""
You are an expert electrical engineer and SPICE modeling specialist.
You are part of a multi-step pipeline that converts schematic images into an
executable SPICE netlist.

PIPELINE OVERVIEW
-----------------
1. Function 1 (already completed) segments a large schematic into smaller
   sub-circuit images. Each image you receive is one of these subcircuits.
2. Function 2 (your role) converts EACH subcircuit image into an accurate SPICE
   subcircuit description.
3. Downstream agents may later refine or replace behavioral models, but they rely
   on your SPICE interface and connectivity being completely accurate.

YOUR ROLE
---------
Given ONE schematic subcircuit image:
- Understand the electrical structure and connectivity.
- Identify components (resistors, capacitors, diodes, BJTs, MOSFETs, passives,
  connectors, simple sources, etc.).
- Identify integrated circuits (ICs) and complex multi-pin devices.
- Produce a SPICE subcircuit (.SUBCKT) that captures ALL external behavior
  interfaces (pins and nets) and basic internal connections (for discrete parts).

IC HANDLING (VERY IMPORTANT)
----------------------------
If the schematic contains any IC or complex multi-pin device:
- DO NOT attempt to model its internal behavior.
- Instead, create a **stub SPICE subcircuit** for each IC.
- The stub MUST strictly maintain the correct pin interface:
  - Use a .SUBCKT declaration with all pins in a clear, consistent order.
  - The pin list and count must reflect the schematic as accurately as possible.
  - Follow the PIN NAMING CONVENTION described below.
  - Example pattern (illustrative only):

        * IC U1: some controller
        .SUBCKT U1_stub VDD_30 GND_15 IN_3 OUT_4 FB_10 NC1_27 NC2_28
        * Internal behavior intentionally omitted; to be filled by another agent.
        .ENDS U1_stub

- In the main subcircuit, instantiate the stub using the same pin order and
  clear net naming.

PIN NAMING CONVENTION (MANDATORY)
---------------------------------
- For any pin that has BOTH:
  - a pin number N in the schematic, and
  - a pin name (label) NAME in the schematic,
  the corresponding SPICE node name MUST be:

      NAME_N

  where N is the pin number.
- Example:
  - If the schematic shows pin number 30 with label VDD, the SPICE node name
    MUST be:

      VDD_30

- Apply this convention consistently for:
  - Subcircuit interface pins (ports) derived from connector or IC pins.
  - Pins of IC stub subcircuits.
- If a pin has a number but no clear name, you may use a generic base like
  PIN or NET, e.g.:

      PIN_30   or   NET_30

  but keep this consistent within the subcircuit.

SPICE SUBCIRCUIT REQUIREMENTS
------------------------------
- Wrap the entire subcircuit logic in a .SUBCKT / .ENDS pair.
- Use a meaningful subcircuit name derived from the image or reference designator,
  e.g. SUBCKT SUBCKT_<IMAGE_NAME> or similar.
- Include all external terminals (nets that connect to other parts of the global
  schematic) as subcircuit ports, and apply the PIN NAMING CONVENTION when
  those ports correspond to numbered pins.
- For discrete components:
  - Use standard SPICE primitives: R, C, L, D, Q, M, V, I, etc.
  - Use reasonable generic models if needed, but keep things simple.
- Use comments liberally to explain assumptions, unclear symbols, or inferred values.
- If any value is unreadable or ambiguous, make a clear, commented assumption:

      * R5 value unreadable; assumed 10k
      R5 NET_A NET_B 10k

NET NAMING & CONNECTIVITY
-------------------------
- Preserve any visible net labels from the schematic when possible (e.g. VCC,
  GND, OUT, IN+, IN-, VS, VBAT).
- Clearly differentiate between different ground symbols if necessary (e.g. GND,
  AGND, PGND).
- Ensure that nodes sharing the same visible label in the schematic are tied
  together in SPICE.
- When a net corresponds directly to a numbered pin, prefer the PIN NAMING
  CONVENTION (NAME_N) for the external interface, and use comments if you need
  to relate that to internal net labels.

ITERATIVE REFINEMENT
--------------------
Within a single task for a given image, follow an internal iterative process:
1. Draft an initial SPICE subcircuit.
2. Self-review for:
   - Missing components.
   - Incorrect or missing pins for ICs.
   - Floating nodes that should not be floating.
   - Obvious syntax errors.
3. Refine and correct until the SPICE is internally consistent and ready for use.
4. Only then write the final SPICE subcircuit to the requested output file.

OUTPUT & FILE SAVING
--------------------
- The workspace root is the directory where this Python script is executed.
- Save the final SPICE subcircuit for each image into the directory:
      {spice_output_dir}
  relative to the workspace.
- Use a file name based on the image stem (e.g. image 'subckt_003.png' ->
  '{spice_output_dir}/subckt_003.cir').
- The content of each .cir file should be ONLY the SPICE subcircuit and comments.
- In your chat messages, keep prose minimal. The main artifact is the .cir file.

GENERAL STYLE
-------------
- Be precise, concise, and engineering-focused.
- Prefer correctness and clarity over fancy formatting.
- If something is genuinely ambiguous, call it out in comments near the
  corresponding SPICE elements.
    """.strip()

# ---------------------------------------------------------------------------
# Helper: discover subcircuit images to process
# ---------------------------------------------------------------------------

def list_segmented_images(images_dir: Path) -> List[Path]:
    """
    Return a sorted list of image paths in the given directory.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Segmented images directory not found: {images_dir}")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"No images found in directory: {images_dir}")
    return images


# ---------------------------------------------------------------------------
# Main pipeline: function 2
# ---------------------------------------------------------------------------

def run_spice_extraction_pipeline(
    segmented_images_subdir: str = "schematic_subcircuits",
    spice_output_subdir: str = "spice_subcircuits",
):
    """
    Function 2: Iterate over segmented images, call the vision LLM agent for each,
    and instruct it to produce SPICE subcircuits saved in a dedicated directory.

    Parameters
    ----------
    segmented_images_subdir : str
        Directory (relative to this script) containing segmented subcircuit images.
    spice_output_subdir : str
        Directory (relative to this script) where .cir files should be saved.
    """
    # ----------------------------------------------------------------------
    # Resolve paths
    # ----------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    images_dir = script_dir / segmented_images_subdir
    spice_dir = script_dir / spice_output_subdir

    # Ensure output directory exists (agent will also be told to use this)
    spice_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Segmented images directory: {images_dir}")
    logger.info(f"SPICE output directory:     {spice_dir}")

    images = list_segmented_images(images_dir)
    logger.info(f"Found {len(images)} segmented images")

    # ----------------------------------------------------------------------
    # Configure LLM (vision-capable)
    # ----------------------------------------------------------------------
    api_key = os.getenv("LLM_API_KEY")
    assert api_key is not None, "LLM_API_KEY environment variable is not set."

    model = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
    base_url = os.getenv("LLM_BASE_URL")

    llm = LLM(
        usage_id="vision-llm",
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
    )
    assert llm.vision_is_active(), "The selected LLM model does not support vision input."

    cwd = os.getcwd()

    # Tools: terminal (for mkdir, etc.), file editor (for writing .cir files), task tracker
    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ],
    )

    system_prompt = build_system_prompt(spice_output_subdir)

    # ----------------------------------------------------------------------
    # Process each image independently
    # ----------------------------------------------------------------------
    for img_path in images:
        logger.info(f"Processing image: {img_path.name}")

        # Collect LLM messages for inspection/debug
        llm_messages = []

        def conversation_callback(event: Event) -> None:
            if isinstance(event, LLMConvertibleEvent):
                llm_messages.append(event.to_llm_message())

        # Fresh conversation per image (clean context)
        conversation = Conversation(
            agent=agent,
            callbacks=[conversation_callback],
            workspace=cwd,
        )

        # 1) Send system instructions (pipeline + SPICE rules)
        conversation.send_message(
            Message(
                role="user",
                content=[TextContent(text=system_prompt)],
            )
        )

        # 2) Send user task + image
        image_uri = img_path.resolve().as_uri()  # file:// URL for local image

        image_stem = img_path.stem
        target_spice_path = f"{spice_output_subdir}/{image_stem}.cir"

        user_instruction = f"""
You are now processing ONE schematic subcircuit image: {img_path.name}.

TASK
----
1. Analyze the attached image and understand the subcircuit.
2. Construct an accurate SPICE .SUBCKT for this subcircuit, following the system
   instructions you received.
3. Make sure any ICs are represented as stub subcircuits with accurate pin
   interfaces only. Internal behavior is omitted.
4. Run your own internal iterative refinement (draft → self-review → correction).
5. When you are satisfied with the result, write the final SPICE subcircuit into:

    {target_spice_path}

   relative to the workspace root, using the FileEditorTool (and TerminalTool
   if needed to create the directory).

IMPORTANT
---------
- The file {target_spice_path} should contain ONLY SPICE code and comments.
- In this chat response, briefly confirm the file path and any key assumptions,
  but do NOT dump the entire SPICE code here; rely on the .cir file instead.
        """.strip()

        conversation.send_message(
            Message(
                role="user",
                content=[
                    TextContent(text=user_instruction),
                    ImageContent(image_urls=[image_uri]),
                ],
            )
        )

        # 3) Run the conversation for this image
        conversation.run()

        # Optional: minimal summary / debug print
        print("=" * 80)
        print(f"Finished processing image: {img_path.name}")
        print(f"Expected SPICE file: {target_spice_path}")
        print("Got the following LLM messages (truncated):")
        for i, message in enumerate(llm_messages):
            msg_str = str(message)
            print(f"Message {i}: {msg_str[:300].replace('\\n', ' ')}")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Cost report (overall)
    # ------------------------------------------------------------------
    cost = llm.metrics.accumulated_cost
    print(f"TOTAL_PIPELINE_COST: {cost}")



from pathlib import Path
from google.cloud import storage


def upload_image_to_gcs(
    local_path: Path,
    bucket_name: str,
    prefix: str = "schematic_subcircuits/",
    make_public: bool = False,
) -> str:
    """
    Upload a local image to a GCS bucket and return a URL usable by Vertex AI.

    If make_public=False (default), returns a 'gs://...' URI, which Gemini/Vertex
    can usually consume directly.

    If make_public=True, it also makes the object world-readable and returns the
    HTTPS URL instead.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # e.g. "schematic_subcircuits/subckt_003.png"
    blob_name = f"{prefix}{local_path.name}"
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(str(local_path))

    if make_public:
        blob.make_public()
        return blob.public_url  # https://storage.googleapis.com/...

    # Default: return GCS URI
    return f"gs://{bucket_name}/{blob_name}"


def run_spice_extraction_pipeline_gcs_bucket(
    segmented_images_subdir: str = "schematic_subcircuits",
    spice_output_subdir: str = "spice_subcircuits",
    gcs_bucket_name: str = "vhl",  # <<< set this
    gcs_prefix: str = "schematic_subcircuits/",       # folder-ish prefix in bucket
):
    """
    Function 2: Iterate over segmented images, upload them to GCS so Vertex AI can
    see them, call the vision LLM agent for each, and instruct it to produce
    SPICE subcircuits saved locally.
    """
    # ----------------------------------------------------------------------
    # Resolve local paths
    # ----------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    images_dir = script_dir / segmented_images_subdir
    spice_dir = script_dir / spice_output_subdir

    spice_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Segmented images directory: {images_dir}")
    logger.info(f"SPICE output directory:     {spice_dir}")

    images = list_segmented_images(images_dir)
    logger.info(f"Found {len(images)} segmented images")

    # ----------------------------------------------------------------------
    # Configure LLM (vision-capable)
    # ----------------------------------------------------------------------
    api_key = os.getenv("LLM_API_KEY")
    assert api_key is not None, "LLM_API_KEY environment variable is not set."

    model = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
    base_url = os.getenv("LLM_BASE_URL")

    llm = LLM(
        usage_id="vision-llm",
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
    )
    assert llm.vision_is_active(), "The selected LLM model does not support vision input."

    cwd = os.getcwd()

    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ],
    )

    system_prompt = build_system_prompt(spice_output_subdir)

    # ----------------------------------------------------------------------
    # Process each image independently
    # ----------------------------------------------------------------------
    for img_path in images:
        logger.info(f"Processing image: {img_path.name}")

        # 1) Upload to GCS and get a URL/URI visible to Vertex AI  ### NEW
        gcs_url = upload_image_to_gcs(
            local_path=img_path,
            bucket_name=gcs_bucket_name,
            prefix=gcs_prefix,
            make_public=False,  # or True if you prefer public HTTPS URLs
        )
        logger.info(f"Uploaded {img_path.name} to {gcs_url}")
        
        # Collect LLM messages for inspection/debug
        llm_messages = []

        def conversation_callback(event: Event) -> None:
            if isinstance(event, LLMConvertibleEvent):
                llm_messages.append(event.to_llm_message())

        conversation = Conversation(
            agent=agent,
            callbacks=[conversation_callback],
            workspace=cwd,
        )

        # System-level instructions
        conversation.send_message(
            Message(
                role="user",
                content=[TextContent(text=system_prompt)],
            )
        )

        image_stem = img_path.stem
        target_spice_path = f"{spice_output_subdir}/{image_stem}.cir"

        user_instruction = f"""
You are now processing ONE schematic subcircuit image: {img_path.name}.

TASK
----
1. Analyze the attached image and understand the subcircuit.
2. Construct an accurate SPICE .SUBCKT for this subcircuit, following the system
   instructions you received.
3. Make sure any ICs are represented as stub subcircuits with accurate pin
   interfaces only. Internal behavior is omitted.
4. Run your own internal iterative refinement (draft → self-review → correction).
5. When you are satisfied with the result, write the final SPICE subcircuit into:

    {target_spice_path}

   relative to the workspace root, using the FileEditorTool (and TerminalTool
   if needed to create the directory).

IMPORTANT
---------
- The file {target_spice_path} should contain ONLY SPICE code and comments.
- In this chat response, briefly confirm the file path and any key assumptions,
  but do NOT dump the entire SPICE code here; rely on the .cir file instead.
        """.strip()

        # Use GCS URL instead of file:// URI  ### NEW
        conversation.send_message(
            Message(
                role="user",
                content=[
                    TextContent(text=user_instruction),
                    ImageContent(image_urls=[gcs_url]),
                ],
            )
        )

        conversation.run()

        print("=" * 80)
        print(f"Finished processing image: {img_path.name}")
        print(f"Expected SPICE file: {target_spice_path}")
        print("Got the following LLM messages (truncated):")
        for i, message in enumerate(llm_messages):
            msg_str = str(message)
            print(f"Message {i}: {msg_str[:300].replace('\\n', ' ')}")
        print("=" * 80)

    cost = llm.metrics.accumulated_cost
    print(f"TOTAL_PIPELINE_COST: {cost}")


if __name__ == "__main__":
    # You can tweak these to match your function 1 output layout.
    run_spice_extraction_pipeline_gcs_bucket(
        segmented_images_subdir="../function_1/schematic_subcircuits",
        spice_output_subdir="spice_subcircuits",
    )

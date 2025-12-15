# Schematic-to-SPICE Conversion Pipeline

This document outlines the three-stage architecture for converting high-resolution circuit schematics into modular, simulation-ready SPICE models. The pipeline moves from visual segmentation to code generation, concluding with an automated audit.

```mermaid
graph LR
    A[Original Schematic] -->|Function 1| B(Sub-circuit Images)
    B -->|Function 2| C(Draft SPICE Code)
    C -->|Function 3| D[Verified SPICE Model]
    A -.->|Ref Context| D
```

-----

## Function 1: Semantic Segmentation (Visual Pre-processing)

**Objective:** Divide complex, high-resolution schematics into manageable, self-contained visual blocks.

  * **Type:** Deterministic ML Pipeline (Non-Agentic).
  * **Core Technology:** **SAMv3** (Segment Anything Model) with Promptable Context Segmentation.
  * **Workflow:**
    1.  Ingests the raw high-resolution schematic image.
    2.  Applies SAMv3 using specific prompt keywords: `"Circuit"`, `"Schematic"`.
    3.  Extracts bounding boxes for functional clusters (e.g., Power Supply block, MCU block).
    4.  Crops and saves these segments as individual image files in local storage.
  * **Key Output:** A directory of segmented sub-circuit images (PNG/JPG).

-----

## Function 2: SPICE Extraction Agent (Generation)

**Objective:** Convert visual sub-circuits into valid SPICE netlists with accurate connectivity and component values.

  * **Type:** Agentic Loop (OpenHands SDK).
  * **Core Technology:** Vision-Language Models (e.g., Gemini 2.5 Flash, Claude 3.5 Sonnet).
  * **Infrastructure:** Uses **Google Cloud Storage (GCS)** to host images, providing accessible URIs for the multimodal LLM.
  * **Workflow:**
    1.  **Upload:** Agent automatically uploads the local segmented image to GCS.
    2.  **Analysis:** The Vision LLM scans the image for components, wires, and text labels.
    3.  **Stubbing:** Complex ICs are identified and converted into `.SUBCKT` stubs (preserving pin interfaces) rather than attempting to model internal logic.
    4.  **Self-Correction:** The agent drafts the code, reviews it for syntax errors or missing pins, and iterates internally.
    5.  **File Generation:** Writes a `.cir` file containing the draft SPICE code.
  * **Key Output:** Draft SPICE Sub-circuit files (`.cir`).

-----

## Function 3: The "Reviewer" Agent (Auditing)

**Objective:** Validate draft SPICE code against the original visuals to detect hallucinations, label blindness, and connectivity errors.

  * **Type:** Agentic Loop (Actor-Critic Pattern).
  * **Core Technology:** High-Reasoning LLM (e.g., Gemini 1.5 Pro).
  * **Workflow:**
    1.  **Dual-Context Loading:** The agent loads the **Draft SPICE Code** + **Segmented Image** + **Original Full Schematic** (for global net context).
    2.  **Visual Audit:** Performs a "Trace & Verify" routine:
          * **Label Check:** Ensures visible text labels (e.g., `BAT`, `VCC`) are explicitly used in the netlist, correcting generic hallucinations (e.g., `NET01`, `PWR_R5_OUT`).
          * **Topology Check:** Verifies that component connections match the visual lines.
    3.  **In-Place Correction:** Uses file editing tools to surgically fix errors in the `.cir` file without rewriting valid code.
  * **Key Output:** Verified, high-fidelity SPICE files ready for final integration.

### Pydoc
Function 3: SPICE Netlist Review & Correction Agent.

This agent acts as a "Visual Netlist Auditor". It takes the SPICE code generated
by Function 2, compares it against the source images (both segmented and full),
detects hallucinations (like label blindness), and applies corrections in-place.

This version:
- Reads *.cir files from GCS.
- Downloads them into a temporary local directory for the agent to edit.
- After review, compares local vs original GCS content.
- If changed, uploads the reviewed file & a diff report to a versioned
  hierarchy in GCS, without overwriting the original.


### Decision on Function 3(Review agent)
Implemented and tested basic review agent workflow

First impressions:

1. Review agent is not capturing wrong mappings as expected. 

	- Tried with Gemini-2.5-flash and Gemini-3.0-pro. 

	- Flash model failed to identify the missing node connections for BAT. It could resolve issues related to R121. But differential changes are minimal. 

	- Gemini-3.0-pro failed to respond properly for the above example image.

02_s27 : This sub-circuit belongs to the BQ79616 ic. There are few mistakes(ones mentioned above) in the extracted SPICE code for this module. Yet review agent failed to identify those mistakes, rather Gemini-3-pro response was not proper. 

01_s39 : Minor updates after review agent. No breaking errors detected

03_s40 : Schematic contain confusing mistake. Resolved the error originated from this mistake. Not major.

04_s39 : Review agent removed all the pin numbering. Undesirable outcome. Happened because review agent is unaware of pin conventions

05_s30 : Review agent resolved one major mistake in the SPICE code. R119 resistor connections were wrong. Agent corrected this. But agent modified pin names of ISO07342 ic without any strong reason

* ** Identified one key missing instruction as part of the Schematic extraction prompt. Right now, we don't specify how node names has to be defined. Since the schematic focuses on system level view, we should consider net names which are present in the schematic nets as the first class persons. Pin names as part of IC definition should be considered as the last resort. If no net name is available for a node, then only use IC pin names as node names


Based on the above results, we decided to move further without a Review agent at this stage. Most of the issues related to the SPICE extraction may get resolved as we could start using Gemini-3-flash(once it releases). Spending extra time and resources for a review agent at this stage doesn't seem to give much differential gain. Hence proceeding without review agent. Code will be archived for later use, if we encounter a strong use case for the review agent.


# Function 4: Integration agent

## Pydoc
Combine multiple subcircuits into a single top-level SPICE netlist.

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

# Function 5: Circuitron integration

## Pydoc
Convert SPICE code to KiCAD schematic using Circuitron 

Inputs:
- SPICE netlist from integration agent

Task:
- Incrementally construct KiCAD schematic from the SPICE code generated by Integration agent
- Agent uses:
  - Top level SPICE netlist from integration agent
  - Supplimentary information: Sub circuit module SPICE codes from Function 2, Overall schematic image, Sub circuit images

# Function 6: Interactive user interface

- Touch points to each one of functions (1,2,and 4)
  - User should be able to suggest modifications on each of these stages
  - System should have mechanisms to distill inferences from the user suggestions
    - It could be modifying prompt instructions to the agent, Adjusting the hyper parameters(in case of function 1), Updating the context for LLM agents etc.
- Before each stage there shoud be provision for user interaction. This help to avoid error propagation from one stage to another
  - After Function 1 user should be able to remove un-necessary bounding boxes. For now we avoid provision to add new bounding boxes. Instead we set the confidence threshold of SAMv3 model to very low, so that model extract maximum bounding boxes. 
  - After Function 2, user can suggest modifications in the generated SPICE code. SPICE code will be presented to the user as an ideal schematic, instead of raw SPICE statements. User should be able to suggest modifications in text. Later we will add functionality to modify connections through GUI
  - Function 4 should be an interactive agent. Agent should prepare overall schematic and present it to the user. User can suggest modifications in text. Same pipeline used in previous stage can be reused here. Backend agent implementation differentiates Function 4 from Function 3.
- In this MVP, we avoid Function 3 altogether. Instead we introduce human in loop. Every human-agent interactions will be documented. Later we can use this data to train a review agent.
  - This review agent will augment exact same touchpoints as the human in loop system of our MVP1.

## UI
- For now, text is the only interface through which user can control behaviour of the system. Any modification has to be conveyed through text. 


# Required functionalities to achieve the objectives
1. Functions 1,2,4
  - Functions 3 and 5 are not part of the first milestone
  - If system can generate accurate SPICE model for the given schematic image, Milestone 1 is completed. 
2. Robust pipeline to convert SPICE code to Schematic image(raster or vector). (Options are ordered based on preference)
  1. Option 1 : Dedicated library which converts SPICE code to schematic image; Explore if any such python libraries are available 
  2. Option 2 : mermaid graph
  3. Option 3 : Vector graphics
  4. Option 4 : Image generation models
  - if we could achieve this, we might be able to bypass circuitron altogether
  - lets call this "Sub Function 1"


# Sub Function 1: Design
- Inputs:
  - SPICE code
- Output:
  - Schematic

- Functionality: 
  - Convert SPICE code to standardized schematic 
  - Pins, nets and connections should be represented with deterministic accuracy. Results should be reproducible. ie same SPICE code always returns same schematic
  - Topolgy, placement of components, and visual pleaseness are of lowest priority. It is always preferred to have minimum number of wire crossings.
  - Output can be any of the following format
    - Some graphical format which can be easily representable through text
    - Mermaid graph
    - Vector graphics
===

# Agent 1 

import os
import sys

# Ensure agent-sdk root is in the python path
sdk_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sdk_root not in sys.path:
    sys.path.insert(0, sdk_root)

# Add vision_agent directory to sys.path to allow importing it directly
vision_agent_dir = os.path.join(sdk_root, "examples", "06_custom_examples", "vision_agent")
if vision_agent_dir not in sys.path:
    sys.path.insert(0, vision_agent_dir)

import vision_agent

def main():
    target_image = "examples/06_custom_examples/vision_agent/guage.jpeg"
    question = "What number is the gauge showing?"
    
    print(f"Starting Vision Investigation Agent on {target_image}...")
    print(f"Question: {question}")
    
    vision_agent.run_vision_agent(target_image, question)

if __name__ == "__main__":
    main()

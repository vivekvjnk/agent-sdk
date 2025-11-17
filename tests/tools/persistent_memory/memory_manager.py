# TODO : Move following test code to proper test directory
# if __name__ == "__main__":
#     # --- Setup ---
#     TEMP_DIR = Path("./temp_memory_test")
#     FILE_NAME = "agent_long_term.txt"
#     MEMORY_PATH = TEMP_DIR / FILE_NAME

#     print("--- Memory System Test ---")
#     if TEMP_DIR.exists():
#         # Clean up existing files
#         for f in TEMP_DIR.glob('*'):
#             f.unlink()
    
#     # --- 1. Singleton Initialization ---
#     print("\n1. Testing Singleton Initialization:")
    
#     # First instantiation
#     manager1 = PersistentMemoryManager(base_path=TEMP_DIR, file_name=FILE_NAME)
#     manager1_id = id(manager1)
#     print(f"Manager 1 ID: {manager1_id}")
    
#     # Second instantiation attempt with different arguments (should return the same object)
#     manager2 = PersistentMemoryManager(base_path=Path("/tmp"), file_name="ignored.txt")
#     manager2_id = id(manager2)
#     print(f"Manager 2 ID: {manager2_id}")

#     print(f"Are manager1 and manager2 the same object? {manager1 is manager2}")
#     if manager1 is manager2:
#         print("PASS: Singleton confirmed.")
#     else:
#         print("FAIL: Singleton failed.")

#     # Check initial file state (should be empty)
#     print(f"\nInitial file created at: {manager1.file_path}")
#     initial_content = content_to_str(manager1.get_content().content)[0]
#     print(f"Initial Content (Manager 1): '{initial_content}'")
    
#     # Create an initial immutable event
#     event_v1 = MemoryEvent(content=manager1.get_content(), memory_file_name=FILE_NAME)
#     print(f"Event V1 created. Content: {event_v1}")

#     # --- 2. Update via Manager and Check Cache ---
#     print("\n2. Testing Manager Update (Internal Consistency):")
#     NEW_CONTENT_1 = "The agent's primary task is to find the bug in the provided Python code."
    
#     manager1.update(NEW_CONTENT_1)
    
#     # Check if cache was eagerly updated (get_content should return new content without disk delay)
#     content_after_update = content_to_str(manager1.get_content().content)[0]
#     print(f"Content After Update: '{content_after_update}'")

#     # Create a new event snapshot
#     event_v2 = MemoryEvent(content=manager2.get_content(), memory_file_name=FILE_NAME) # manager2 is the same object
#     print(f"Event V2 created (New Snapshot). Content: {event_v2}")

#     # Verify Immutability
#     print(f"\nVerifying Immutability:")
#     event_v1_content = content_to_str(event_v1.content.content)[0]
#     event_v2_content = content_to_str(event_v2.content.content)[0]
#     print(f"Event V1 Content (Immutable): '{event_v1_content}'")
#     print(f"Event V2 Content (New Snapshot): '{event_v2_content}'")
    
#     if event_v1_content != event_v2_content and event_v1_content == initial_content:
#         print("PASS: Immutability and Snapshots confirmed.")
#     else:
#         print("FAIL: Immutability check failed.")

#     # --- 3. External Change Detection ---
#     print("\n3. Testing External Change Detection (Mocking File Change):")
#     NEW_CONTENT_EXTERNAL = "Important note: Always use the Python standard library."
    
#     # Directly write to the file (simulating an external tool change)
#     with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
#         f.write(NEW_CONTENT_EXTERNAL)
    
#     # Wait a moment to ensure mtime changes (necessary for some file systems)
#     time.sleep(0.1) 
    
#     # Reload content (should detect the change and load from disk)
#     manager1.reload_content()
    
#     content_after_external_change = content_to_str(manager1.get_content().content)[0]
#     print(f"Content After External Change: '{content_after_external_change}'")

#     if content_after_external_change == NEW_CONTENT_EXTERNAL:
#         print("PASS: External change detection confirmed (Content reloaded).")
#     else:
#         print("FAIL: External change detection failed.")
        
#     # --- Cleanup ---
#     # print("\n--- Cleanup ---")
#     # if MEMORY_PATH.exists():
#     #     MEMORY_PATH.unlink()
#     #     print(f"Removed temporary file: {MEMORY_PATH}")
#     # if TEMP_DIR.exists():
#     #     TEMP_DIR.rmdir()
#     #     print(f"Removed temporary directory: {TEMP_DIR}")
# setup_shared_memory.py
import os
from shared_memory_manager import SharedMemoryManager

# Configuration
CONFIG_FILE = "mem_setup.json"

def main():
    manager = SharedMemoryManager(CONFIG_FILE)

    try:
        # Allocate shared memory as per configuration
        manager.create_memory()

        # Save allocated memory names to a separate file
        manager.save_allocations()
        print(f"Shared memory allocations saved to tmp/mem_setup_alloc.json")

        # Keep the script running
        print("Shared memory created. Press Ctrl+C to terminate.")
        while True:
            pass

    finally:
        manager.cleanup_memory()
        print("Shared memory cleaned up.")

if __name__ == "__main__":
    main()
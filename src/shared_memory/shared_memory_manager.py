# shared_memory_manager.py
from multiprocessing import shared_memory
import numpy as np
import json
import os
# Unregister shared memory from resource tracker
import multiprocessing.resource_tracker

class SharedMemoryManager:
    def __init__(self, config_file, alloc_file="tmp/mem_setup_alloc.json"):
        self.config_file = config_file
        self.alloc_file = alloc_file
        self.shared_memory_objects = {}
        self.dtype_map = {
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "float32": np.float32,
        }

        os.makedirs(os.path.dirname(self.alloc_file), exist_ok=True)

    def load_config(self):
        with open(self.config_file, 'r') as f:
            return json.load(f)

    def save_allocations(self):
        allocations = {name: shm.name for name, shm in self.shared_memory_objects.items()}
        with open(self.alloc_file, 'w') as f:
            json.dump(allocations, f, indent=4)

    def load_allocations(self):
        with open(self.alloc_file, 'r') as f:
            return json.load(f)

    def create_memory(self):
        config = self.load_config()
        for name in list(config.keys()):
            mem = config[name]
            size, dtype = mem["size"], mem["dtype"]
            size_bytes = np.prod(size) * np.dtype(dtype).itemsize
            dtype_resolved = self.dtype_map[dtype]
            shm = shared_memory.SharedMemory(create=True, size=size_bytes)
            multiprocessing.resource_tracker.unregister(shm._name, 'shared_memory')
            np_array = np.ndarray(size, dtype=dtype_resolved, buffer=shm.buf)
            np_array[:] = 0  # Initialize memory with zeros
            self.shared_memory_objects[name] = shm

    def load_memory(self, name, size, dtype):
        allocations = self.load_allocations()
        dtype_resolved = self.dtype_map[dtype]
        shm_name = allocations[name]
        shm = shared_memory.SharedMemory(name=shm_name)
        multiprocessing.resource_tracker.unregister(shm._name, 'shared_memory')
        self.shared_memory_objects[name] = shm  # <- tienilo in vita!
        return np.ndarray(size, dtype=dtype_resolved, buffer=shm.buf), shm

    def cleanup_memory(self):
        for shm in self.shared_memory_objects.values():
            shm.close()
            shm.unlink()
        self.shared_memory_objects.clear()

        if os.path.exists(self.alloc_file):
            os.remove(self.alloc_file)
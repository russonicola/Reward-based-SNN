import h5py
import json
import numpy as np

class HDF5Manager:
    def __init__(self, file_path):
        """
        Initialize the HDF5 file manager.
        Creates the file if it doesn't exist or opens it in append mode.
        """
        self.file_path = file_path
        self.file = h5py.File(file_path, "a")  # Open in append mode
        
        # Create a resizable dataset if it doesn't exist
        if "data" not in self.file:
            self.file.create_dataset(
                "data",
                shape=(0,),  # Initial size of the dataset
                maxshape=(None,),  # Allow unlimited rows
                dtype=h5py.string_dtype(encoding="utf-8")  # Store serialized JSON strings
            )
    
    def append(self, obj_data):
        """
        Append a new object to the dataset.
        obj_data: Dictionary containing object data to be appended.
        """
        dataset = self.file["data"]
        current_size = dataset.shape[0]

        # Resize the dataset to accommodate the new entry
        dataset.resize((current_size + 1,))  # Ensure shape is a tuple
        serialized_data = json.dumps(obj_data)  # Convert and serialize

        # Store the serialized string
        dataset[current_size] = serialized_data
    
    def get_all(self):
        """
        Retrieve all objects in the dataset.
        Returns a list of deserialized objects.
        """
        dataset = self.file["data"]
        return [json.loads(obj) for obj in dataset[:]]
    
    def close(self):
        """Close the HDF5 file."""
        self.file.close()
        
        
        
import json
import os


class JSONLinesManager:
    def __init__(self, file_path):
        """
        Initialize the JSON Lines file manager.
        Creates the file if it doesn't exist.
        """
        self.file_path = file_path
        # Create the file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w'):
                pass

    def append(self, obj_data):
        """
        Append a new object to the JSON Lines file.
        obj_data: Dictionary containing object data to be appended.
        """
        with open(self.file_path, 'a') as f:
            f.write(json.dumps(obj_data) + '\n')  # Write JSON object as a single line

    def get_all(self):
        """
        Retrieve all objects from the JSON Lines file.
        Returns a list of deserialized objects.
        """
        with open(self.file_path, 'r') as f:
            return [json.loads(line) for line in f]

    def clear(self):
        """
        Clear all data in the JSON Lines file.
        """
        with open(self.file_path, 'w'):
            pass  # Truncate the file to make it empty
        
from collections import defaultdict
from typing import List, Union

class AERProcessing:
    def __init__(self, file_path):
        """
        Initialize the JSON Lines file manager.
        """
        self.file_path = file_path
        self.records = []
        self.dtype = np.int32
        
        self.origin_x = 128
        self.origin_y = 128
        self.new_x = 8
        self.new_y = 8
        self.downscaling_factor_x = int(self.origin_x/self.new_x)
        self.downscaling_factor_y = int(self.origin_y/self.new_y)
        self.selected_rows = [1,5]
        
        with open(self.file_path, 'r') as f:
            self.records = [json.loads(line) for line in f]
            
            
    def set_resolution(self, origin_x, origin_y, new_x, new_y, selected_rows):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.new_x = new_x
        self.new_y = new_y
        self.downscaling_factor_x = int(self.origin_x/self.new_x)
        self.downscaling_factor_y = int(self.origin_y/self.new_y)
        self.selected_rows = selected_rows
        
            
    """
    Convert the absolute timings of each record into relative to itself.
    """
    def normalise_time(self):
        for record in self.records:
            data = record['data']
            offset = data[0][0][0] # get current offset
            data[0][0][0] = 0 # set first to 0
            
            for x in range(1, len(data)):
                data[x][0] -= offset
                
    def downscale_packet(self, packet):
        packet[1] = packet[1] // self.downscaling_factor_x
        packet[2] = packet[2] // self.downscaling_factor_y
        
    
    def downscale(self, grouped_record, threshold=10):
        
        counter = 0
        
        data = []
        rows = []
        
        for time_window in grouped_record:
            downscaled_frame = np.zeros((self.new_x, self.new_y), dtype=self.dtype)
            filtered_rows = np.zeros(0, dtype=self.dtype)
    
            for spike in time_window:
                
                if len(spike) == 0:
                    break
                
                # inline downscaling
                self.downscale_packet(spike)
                downscaled_frame[spike[1], spike[2]] += 1
            
            # filter using threshold
            for i in range(self.new_x):
                for j in range(self.new_y):
                    if downscaled_frame[i,j] > threshold:
                        counter+=1
                        
                    downscaled_frame[j,i] = 1 if downscaled_frame[i,j] > threshold else 0
                    
            # select rows
            for i in range(self.new_x):
                if i in self.selected_rows:
                    filtered_rows = np.hstack((filtered_rows, downscaled_frame[i]))
                    
            data.append(downscaled_frame)
            rows.append(filtered_rows)
            
        print(f'pixels bigger than {threshold}: {counter}')
        return data, rows
    
    
    """
    Group and sum spikes that are part of the same time window.
    """   
    def group_by_time(self):
        for record in self.records:
            record['data'] = self.process_aer_data(record['data'])
            

    def process_aer_data(self, packets: List[List[List[Union[int, int, int, int, int]]]]) -> List[List[List[Union[int, int, int, int, int]]]]:
        # Step 1: Flatten the packets, preserving the internal structure
        flat_stream = [subpacket for packet in packets for subpacket in packet]
        
        # Step 2: Normalize timestamps by removing the offset of the first timestamp
        first_timestamp = flat_stream[0][0]  # Assume the first spike in the first subpacket has the earliest timestamp
        for spike in flat_stream:
            spike[0] -= first_timestamp
        
        # Step 3: Group subpackets into 1 ms packets based on normalized timestamps
        grouped_packets = defaultdict(list)
        for spike in flat_stream:
            timestamp_ms = spike[0] // 1000  # Convert microseconds to milliseconds
            grouped_packets[timestamp_ms].append(spike)
        
        # Step 4: Fill in missing milliseconds with empty packets
        min_time_ms = 0
        max_time_ms = max(grouped_packets.keys())
        result = []
        for ms in range(min_time_ms, max_time_ms + 1):
            if ms in grouped_packets:
                result.append(grouped_packets[ms])
            else:
                result.append([[]])  # Represents missing ms with the time in microseconds # was [[[None]]]

        return result
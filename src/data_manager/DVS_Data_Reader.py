from __future__ import print_function

import os
import numpy as np
import cv2
from pyaer.dvs128 import DVS128
from datetime import datetime
import time

from src import SharedMemoryManager
from src import HDF5Manager
from src import JSONLinesManager


# Get the current timestamp
current_timestamp = datetime.now()
# Format the timestamp
formatted_timestamp = current_timestamp.strftime("%Y%m%d_%H%M%S")



# Configuration
CONFIG_FILE = "mem_setup.json"
sm_manager = SharedMemoryManager(CONFIG_FILE)

# Create an instance of the manager
os.makedirs('out', exist_ok=True)
#file_manager = HDF5Manager("out/"+formatted_timestamp+".h5")
file_manager = JSONLinesManager("out/"+formatted_timestamp+".jsonl")

# Initialize the DVS128 device
device = DVS128(noise_filter=True)
device.start_data_stream(max_packet_interval=1000)  # each 1ms
# Load new config
device.set_bias_from_json("scripts/configs/dvs128_config.json")
print(device.get_bias())

# For resizing the OpenCV display window
display_width = 512
display_height = 512

# Create a frame for pixel intensities
frame = np.zeros((128, 128), dtype=np.int16)

# Create a matrix to track the last event time for each pixel
last_event_time = np.zeros((128, 128), dtype=np.float64)

# Discharge time (10ms = 0.01s)
discharge_time = 0.02  # in seconds

# Record Variable
trajectory = {
    'background_color': None,
    'speed': None,
    'trajectory_type': None,
    'position' : None,
    'data': []
}

# Recording Function
def record_data(trajectory):
    pass


try:
    # Load shared memory for reading
    config = sm_manager.load_config()
    mem_config = next(mem for mem in config["shared_memories"] if mem["role"] in ["reader", "shared"])
    shm_array, _ = sm_manager.load_memory(mem_config["name"], tuple(mem_config["size"]), mem_config["dtype"])
        

    while True:
        try:
            status, background_color, speed, trajectory_type, position = shm_array.tolist()[0]
            
            
            current_time = time.time()  # Get current time in seconds
            (pol_events, num_pol_event, special_events, num_special_event) = device.get_event('events')

            # Print event information
            #print(f"Number of polarity events: {num_pol_event}")
            #print(f"Number of special events: {num_special_event}")

            # Process incoming events
            if pol_events is not None:
                # Check if the trajectory must be recorded
                if status == 1:
                    trajectory['data'].append(pol_events.tolist())
                else:
                    if status == 2:
                        if len(trajectory['data']) > 0:
                            
                            trajectory['background_color'] = background_color
                            trajectory['speed'] = speed
                            trajectory['trajectory_type'] = trajectory_type
                            trajectory['position'] = position
                            
                            file_manager.append(trajectory)
                            # store json
                            
                    trajectory = {
                        'background_color': None,
                        'speed': None,
                        'trajectory_type': None,
                        'position' : None,
                        'data': []
                    }
                
                for event in pol_events:
                    x, y, polarity = event[1], event[2], event[3]
                    if polarity == 1:
                        frame[y, x] = 255  # Positive event (red intensity)
                    else:
                        frame[y, x] = -255  # Negative event (blue intensity)
                    last_event_time[y, x] = current_time  # Update the timestamp for this pixel

            # Update the frame with discharge mechanism
            time_since_last_event = current_time - last_event_time
            frame[time_since_last_event > discharge_time] = 0  # Reset pixels that exceeded the discharge time

            # Create a color image with a white background
            color_frame = np.ones((128, 128, 3), dtype=np.uint8) * 255  # White background

            # Map positive events to red (255, 0, 0)
            positive_mask = frame > 0
            color_frame[positive_mask] = [0, 0, 255]

            # Map negative events to blue (0, 0, 255)
            negative_mask = frame < 0
            color_frame[negative_mask] = [255, 0, 0]

            # Resize for better visualization
            resized_frame = cv2.resize(color_frame, (display_width, display_height), interpolation=cv2.INTER_NEAREST)

            # Display the frame
            cv2.imshow('DVS Events Visualization', resized_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
        except KeyboardInterrupt:
            print("Shutting down...")
            device.shutdown()
            break

    
finally:
    print("Reader client shutting down.")
    device.shutdown()
    #file_manager.close()
    cv2.destroyAllWindows()
    
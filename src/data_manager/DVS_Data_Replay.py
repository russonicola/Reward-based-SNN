from __future__ import print_function

import os
import numpy as np
import cv2
from datetime import datetime
import time

from Data_Collector import AERProcessing

JSON_PATH = 'out/20250123_143804.jsonl'

preprocessor = AERProcessing(JSON_PATH)
trajectories = preprocessor.group_by_time()

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


try:
    for trajectory in trajectories:
        for pol_events in trajectory:
            try:
                start_time = time.perf_counter()

                # Process events
                for event in pol_events:
                    if event[0] is not None:
                        x, y, polarity = event[1], event[2], event[3]
                        frame[y, x] = 255 if polarity == 1 else -255
                        last_event_time[y, x] = start_time

                # Update the frame
                time_since_last_event = time.perf_counter() - last_event_time
                frame[time_since_last_event > discharge_time] = 0

                # Create a colour image and resize
                color_frame = np.ones((128, 128, 3), dtype=np.uint8) * 255
                color_frame[frame > 0] = [0, 0, 255]
                color_frame[frame < 0] = [255, 0, 0]
                resized_frame = cv2.resize(color_frame, (display_width, display_height), interpolation=cv2.INTER_NEAREST)

                # Display
                cv2.imshow('DVS Events Visualization', resized_frame)

                # Align timing
                elapsed = time.perf_counter() - start_time
                time.sleep(max(0, 0.001 - elapsed))  # Ensure 1ms time window

                # Exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                

            except KeyboardInterrupt:
                print("Shutting down...")
                break
finally:
    print("Reader client shutting down.")
    cv2.destroyAllWindows()
    
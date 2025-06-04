import threading
import os
import json
import sys
import requests
import signal
import time
import copy

import numpy as np

from datetime import datetime
from pyaer.dvs128 import DVS128
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider, Toggle, Select, LinearColorMapper, Button, TextInput, Div
from bokeh.layouts import row, column
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.document import without_document_lock
from tornado.ioloop import IOLoop


from data_manager import JSONLinesManager
from shared_memory import SharedMemoryManager

# Load configuration file
with open('config.json', 'r') as f:
    config = json.load(f)


# Get the current timestamp
current_timestamp = datetime.now()
# Format the timestamp
formatted_timestamp = current_timestamp.strftime("%Y%m%d_%H%M%S")


# Configuration
CONFIG_FILE = "mem_setup.json"
sm_manager = SharedMemoryManager(CONFIG_FILE)


# Load shared memory for writing
configs_sm = sm_manager.load_config()

mem_name_shm_input = 'shm_input'
mem_config_shm_input = configs_sm[mem_name_shm_input]
shm_input_size = mem_config_shm_input["size"]
shm_input_dtype = mem_config_shm_input["dtype"]
shm_input, _ = sm_manager.load_memory(mem_name_shm_input, tuple(mem_config_shm_input["size"]), mem_config_shm_input["dtype"])

mem_name_shm_servo = 'shm_servo'
mem_config_shm_servo = configs_sm[mem_name_shm_servo]
shm_servo_dtype = mem_config_shm_servo["dtype"]
shm_servo, _ = sm_manager.load_memory(mem_name_shm_servo, tuple(mem_config_shm_servo["size"]), mem_config_shm_servo["dtype"])

mem_name_shm_reward = 'shm_reward'
mem_config_shm_reward = configs_sm[mem_name_shm_reward]
shm_reward_dtype = mem_config_shm_reward["dtype"]
shm_reward, _ = sm_manager.load_memory(mem_name_shm_reward, tuple(mem_config_shm_reward["size"]), mem_config_shm_reward["dtype"])




resolution = config['resolution']
downscaling = config['downscaling']
rows = config['rows']
save_stream = config['save_stream']

downscaling_factor = [
    int(resolution[0]/downscaling[0]), 
    int(resolution[1]/downscaling[1])]


array_size_in = config['input_neurons']
array_size_out = config['output_neurons']
array_size_rew = config['reward_neurons']
excluded_side_pixels = config['excluded_side_pixels']
in_to_out_mapping = config['in_to_out_mapping']

batch_size = config['batch_size']
reading_interval = config['reading_interval']

dtype = np.int32


servo_positions = [1, 0.5, 0, -0.5, -1]



from collections import defaultdict

def pixel_mapping(original_size=128):
    mappings = {}
    block_sizes = [2, 4, 8, 16, 32, 64]  # Different target block sizes

    for block_size in block_sizes:
        scale_factor = original_size // block_size
        original_to_reduced = {}
        reduced_to_original = defaultdict(lambda: {"coordinates": {}, "flat_indices": []})

        for i in range(original_size):
            for j in range(original_size):
                # Calculate the coordinates in the reduced matrix
                reduced_i = i // scale_factor
                reduced_j = j // scale_factor
                
                # Calculate the boundaries of the reduced pixel block in the original matrix
                top_left = (reduced_i * scale_factor, reduced_j * scale_factor)
                bottom_right = ((reduced_i + 1) * scale_factor - 1, (reduced_j + 1) * scale_factor - 1)
                
                # Map original to reduced with boundaries
                original_to_reduced[(i, j)] = {
                    "reduced_coord": (reduced_i, reduced_j),
                    "boundaries": {"top_left": top_left, "bottom_right": bottom_right}
                }
                
                # Map reduced to original with coordinates and flat index
                reduced_to_original[(reduced_i, reduced_j)]["coordinates"][(i, j)] = True
                flat_index = i * original_size + j
                reduced_to_original[(reduced_i, reduced_j)]["flat_indices"].append(flat_index)

        # Add mappings for the current block size
        mappings[block_size] = {
            "original_to_reduced": original_to_reduced,
            "reduced_to_original": dict(reduced_to_original)  # Convert defaultdict to dict for easier handling
        }

    return mappings





# Generate the pixel mapping
PIXEL_MAPPING = pixel_mapping()


# -----------------------------------------------------------------
# --------------------------- SERVER ------------------------------
# -----------------------------------------------------------------



class BokehServerThread(threading.Thread):

    def __init__(self):
        super().__init__()
        
        
        #mem_config = config_shared_memory()
        #self.np_read = mem_config['shm_read']
        #self.np_read_fed = mem_config['shm_read_fed']
        #self.shm_read_gen_pos = mem_config['shm_read_gen_pos']
        #self.np_read_rew = mem_config['shm_read_rew']
        #self.np_write = mem_config['shm_write']
        #self.np_write_volt = mem_config['shm_write_volt']
        #self.np_pos = mem_config['shm_write_pos']
        
        # write dvs data to the SNN model
        #self.read_shm = shared_memory.SharedMemory(name=self.np_read)
        # read feedback from VMCU
        #self.read_shm_fed = shared_memory.SharedMemory(name=self.np_read_fed)
        # read prediction from SNN model
        #self.read_shm_gen_pos = shared_memory.SharedMemory(name=self.shm_read_gen_pos)
        # write feedback to the SNN model
        #self.read_shm_rew = shared_memory.SharedMemory(name=self.np_read_rew)
        
        #self.write_shm = shared_memory.SharedMemory(name=self.np_write)
        #self.write_shm_volt = shared_memory.SharedMemory(name=self.np_write_volt)
        
        # read prediction from SNN model
        #self.pos_shm = shared_memory.SharedMemory(name=self.np_pos)
        
        
        # Record Variable
        self.trajectory = {
            'background_color': None,
            'speed': None,
            'trajectory_type': None,
            'position' : None,
            'data': [],
            'rows': []
        }
        

        self.device = DVS128(noise_filter=True)
        self.device.set_bias_from_json("dvs128_config.json")

        self.img = np.zeros((128,128))
        self.img_red = np.zeros((128,128))
        self.img_pos = np.zeros((1,array_size_in), dtype=dtype)
        self.img_res = np.zeros((1,array_size_out), dtype=dtype)
        self.scatter_res = {'x': [], 'y': []}
        self.voltage_res = {'x': [], 'y': []}
        self.timer = time.time()
        self.timer_rew = time.time()
        
        self.batch_input = []
        self.feedback_signal = False
        self.gen_pos_feedback = False
        self.datastream = []
    

    def reduce_matrix_dimensionality(self, matrix, threshold, new_size=(32, 32)):
        old_size = matrix.shape
        block_size = old_size[0] // new_size[0], old_size[1] // new_size[1]
        
        reduced_matrix = np.zeros(new_size, dtype=matrix.dtype)
        
        for i in range(new_size[0]):
            for j in range(new_size[1]):
                # Extract the block from the original matrix
                block = matrix[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]]
                
                # Calculate the mode and its count
                #block_mode, count = mode(block, axis=None)

                block = block.reshape((1,block_size[0]*block_size[1]))[0]
                block_mode = block[block.argmax()]
                block_sum = block.sum()

                # Check the density condition (>50%)
                if block_sum >= (block_size[0] * block_size[1]) * (threshold / 100):
                    reduced_matrix[i, j] = block_mode
        
        return reduced_matrix
    
    
    
    @without_document_lock
    def modify_doc(self, doc):
        custom_palette = ['#ffffff', '#0015ff', '#ff0000']
        color_mapper = LinearColorMapper(palette=custom_palette, low=0, high=2)
        color_mapper_pos = LinearColorMapper(palette=custom_palette, low=0, high=1)

        source = ColumnDataSource(data=dict(image=[self.img]))
        source2 = ColumnDataSource(data=dict(image=[self.img_pos]))
        source3 = ColumnDataSource(data=self.scatter_res)
        source4 = ColumnDataSource(data=self.scatter_res)

        p = figure(width=400, height=400, x_range=(0, 127), y_range=(127, 0), title="DVS Input")
        p.image(image='image', x=0, y=0, dw=128, dh=128, source=source, color_mapper=color_mapper) # , palette=custom_palette

        p_pos = figure(width=400, height=100, x_range=(0, 8), y_range=(0, 1), title="Positions")
        p_pos.image(image='image', x=0, y=0, dw=8, dh=1, source=source2, palette="Spectral11") #palette="Spectral11"
        
        p_res = figure(title="Resulting Spikes", x_axis_label='X', y_axis_label='Y', width=600, height=300, y_range=(-1, 8))
        p_res.scatter(x='x', y='y', source=source3, size=5)

        p_volt = figure(title="Conductance", x_axis_label='X', y_axis_label='Y', width=600, height=300, y_range=(-0.1, 0.6))
        p_volt.line(x='x', y='y', source=source4)

        div_title = Div(text="Log:",width=400, height=150) # 550 to cover also buttons
        div_log = Div(text="",width=400, height=150)
        
        sizes = ['2','4','8','16','32','64','128']
        select_size = Select(value=sizes[2], options=sizes, title="Size")
        slider_threshold = Slider(start=0.0, end=100.0, value=50.0, step=1.0, title="Threshold", width=150)
        slider_accumulator = Slider(start=0.001, end=0.01, value=0.005, step=.001, title="Accumulator DVS", width=150) # was 0.01 0.05 0.01
        slider_accumulator_res = Slider(start=0, end=1.0, value=0.5, step=.1, title="Delay results", width=150)
        slider_accumulator_threshold = Slider(start=0, end=100, value=30, step=5, title="Threshold positions", width=150)
        slider_rotate = Slider(start=0, end=270, value=90, step=90, title="Rotate", width=150)
        toggle_positive = Toggle(label="Positive only", button_type="success", active=False)
        toggle_noise = Toggle(label="Noise on", button_type="success", active=True)
        toggle_video = Toggle(label="Video on", button_type="success", active=True)

        duration = TextInput(value="1", title="Duration (s):")
    
        def update_ranges(attr, old, new):
            p.x_range.start = 0
            p.x_range.end = int(new)
            
            p.y_range.start = int(new)
            p.y_range.end = 0

        #select_size.on_change('value', update_ranges)
        
        def preprocess(pol_events, positive_only=False, noise_on=True, rotation=90):
            
            downscaled_frame = np.zeros((downscaling[0], downscaling[1]), dtype=dtype)
            
            for t,x,y,a,b in pol_events:
                if positive_only:
                    if a == 1:
                        evt = 1 # was 100
                    else:
                        evt = 0
                else:
                    evt = 1 if a == 0 else 2 # was 50 and 100

                if noise_on and b == 1:
                    evt=0


                if evt > 0:
                    if rotation == 0:
                        self.img[x,y]=evt
                        xx = x
                        yy = y
                    elif rotation == 90:
                        self.img[y,x]=evt
                        xx = y
                        yy = x
                    elif rotation == 180:
                        self.img[-x,y]=evt
                        xx = -x
                        yy = y
                    elif rotation == 270:
                        self.img[-y,-x]=evt
                        xx = -y
                        yy = -x
                        
                    # inline downscaling
                    downscaled_x = x // downscaling_factor[0]
                    downscaled_y = y // downscaling_factor[1]
                    downscaled_frame[downscaled_y, downscaled_x] += 1
                    
            return downscaled_frame
        
        
        def filter_rows(downscaled_frame):
            # filter downscaled applying a threshold >10 spikes -> 1 else 0    
            flat_matrix = downscaled_frame.flatten()
            #print(flat_matrix)
            filtered_elements = [1 if i > 10 else 0 for i in flat_matrix]
            #print(filtered_elements)
            
            # filter
            filtered_rows = []
            for row in rows:
                start_idx = row * downscaling[1]
                end_idx = start_idx + downscaling[1]
                filtered_rows.extend(filtered_elements[start_idx:end_idx])
                
            return filtered_rows
            
        

        def update():
            
            size = int(select_size.value)
            threshold = slider_threshold.value
            accumulate = slider_accumulator.value
            accumulate_res = slider_accumulator_res.value
            accumulator_threshold = slider_accumulator_threshold.value
            rotation = slider_rotate.value
            positive_only = toggle_positive.active
            noise_on = toggle_noise.active
            show_img = toggle_video.active
            
            # initialize to 0
            ds = [0 for _ in range(array_size_in)] + [0]
            
            
            # Read shared memory from the ball generator
            #status, background_color, speed, trajectory_type, position = shm_array.tolist()[0]
            #print(f'status {status} {background_color} {speed} {trajectory_type} {position}')
            
            #if status == 1:
            #    print(f'status {status}')
                
            current_time = time.time()  # Get current time in seconds
            events = self.device.get_event('events')
            
            if events is not None: 
                (pol_events, num_pol_event, special_events, num_special_event) = events

                #print(pol_events)
                if pol_events is not None: # store if not None
                    
                    
                    # check reduced dimension and init reduced image
                    self.img_red = np.zeros((size,size))
                    
                    # preprocess DVS input
                    downscaled_frame = preprocess(pol_events)
                    #print(downscaled_frame)
                    
                    # filter specific rows from downlscaled
                    filtered_rows = filter_rows(downscaled_frame)
                    #print(filtered_rows)
                    
                    
        
                    #print(f'elapsed time: {time.time()-current_time}')
                    
                    # SEND VISUAL INPUT TO DU
                    # write batch to the SNN model
                    if len(self.batch_input) == shm_input_size[0]:
                        #shm_input = np.ndarray((batch_size,array_size_in), dtype=shm_input_dtype, buffer=self.read_shm.buf)
                        shm_input[:] = self.batch_input
                        self.batch_input = [filtered_rows]
                    else:
                        self.batch_input.append(filtered_rows)
                        
                    #ds = filtered_rows + [0]
            
            # manage feedback signal from VMCU
            '''
            shared_feedback = np.ndarray((1,array_size_rew+1), dtype=dtype_fed, buffer=self.read_shm_fed.buf)
            
            gen_pos_feedback = np.ndarray((1,array_size_rew+1), dtype=dtype_fed, buffer=self.read_shm_gen_pos.buf)
            
            if bool(gen_pos_feedback[0][0]) != self.gen_pos_feedback:
                # read new data and share it with DU
                #shared_array_feedback = np.ndarray((batch_size,array_size_in), dtype=dtype, buffer=self.read_shm.buf)
                #shared_array_feedback[:] = shared_feedback[:]
                print(gen_pos_feedback[0][1:])
                ds[-1] = 1
                # reset sequence counter
                self.gen_pos_feedback = not self.gen_pos_feedback
            '''  
            

            # Read pixels from original image using mapping and summing them
            flat_img = self.img.flatten()
            for k in list(PIXEL_MAPPING[size]['reduced_to_original'].keys()):
                block = flat_img[PIXEL_MAPPING[size]['reduced_to_original'][k]['flat_indices']]
                if block.sum() > (len(block) * (threshold/100)):
                    self.img_red[k] = 1
                        

            # read here results
            #shared_pos = np.ndarray((1,array_size_out), dtype=dtype, buffer=self.pos_shm.buf)
            #self.img_res += shared_pos
            
            

            if (time.time() - self.timer) > accumulate:
                
                # update image
                if show_img:
                    #img = self.reduce_matrix_dimensionality(self.img, threshold, new_size=(size, size))
                    img = self.img
                    source.data = dict(image=[img])
                    self.img = np.zeros((128,128))

                self.timer = time.time()


                #self.processed_queue.append(self.img_pos)
                img_pos = self.img_red.flatten()
                
                #shared_array = np.ndarray((1,array_size_in), dtype=dtype, buffer=self.read_shm.buf)
                #shared_array[:] = img_pos # maybe it is necessary to check if it is empty and accumulate before empty
                
                
                self.img_red = np.zeros((1,array_size_in))

                #div_log.text = str(img_pos) + '<br>'
                div_log.text = str(self.img_red) + '<br>'
                

                max_evts = self.img_res.max()
                #max_count = self.img_res.count(max_evts)
                max_count = np.sum(self.img_res == max_evts)
                
                '''
                if max_evts > 0 and max_count == 1:
                    servo_pos = int(self.img_res.argmax())
                    #print(self.img_res)
                    # measure response time
                    print('position', servo_pos)
                    
                    try:
                        response = requests.post('http://localhost:5001/prediction', json={'position': servo_pos})
                        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

                        # Check the response status and print the result
                        if response.status_code == 200:
                            data = response.json()
                            print("Response from server:", data)
                        else:
                            print(f"Unexpected status code: {response.status_code}")
                    except requests.exceptions.RequestException as e:
                        print(f"Request failed: {e}")
                        # Handle the error or skip the request
                    
                    source2.data = dict(image=[self.img_pos])
                    div_log.text += str(self.img_res)
                '''

            else:
                self.img_pos = np.zeros((1,array_size_in))
                img_pos = self.img_pos
                #shared_array = np.ndarray((1,array_size_in), dtype=dtype, buffer=self.read_shm.buf)
                self.img_res = np.zeros((1,array_size_out), dtype=dtype)
                
                # if (time.time() - self.timer_res) > accumulate_res:
                #     if len(self.result_queue) > 0:
                #         res = self.result_queue.popleft()
                #         self.scatter_res = {'x': res[0], 'y': res[1]}
                #         source3.data = self.scatter_res
                #     self.timer_res = time.time()
                
                
                
            #shared_array_rew = np.ndarray((1,array_size_rew), dtype=dtype, buffer=self.read_shm_rew.buf)
                
                
            #if (time.time() - self.timer_rew) > 0.010:
            #    
            #    # check http reward
            #    
            #    self.timer_rew = time.time()
            #    shared_array_rew[:] = [1]
            #    
            #else:
            #    shared_array_rew[:] = np.zeros((1,array_size_rew))
                    
                    

                
        c1 = column(p, p_pos, div_log)
        c2 = column(select_size, 
                    slider_accumulator,
                    slider_threshold,
                    slider_accumulator_res, 
                    slider_accumulator_threshold,
                    slider_rotate, 
                    toggle_positive, 
                    toggle_noise,
                    toggle_video)
        c3 = column(p_res, p_volt)
                    
        layout = row(c1,c2,c3)
        doc.add_root(layout)
        doc.add_periodic_callback(update, 1) # each 500ms 10ms prev 0.00001
        doc.add_next_tick_callback(update)
    


    def run(self):
        io_loop = IOLoop.current()
        self.device.start_data_stream(max_packet_interval=reading_interval) # was 1, 1000
        bokeh_app = Application(FunctionHandler(self.modify_doc))
        server = Server({'/monitor': bokeh_app}, io_loop=io_loop, allow_websocket_origin=['*'])
        server.start()
        io_loop.start()


    def stop(self):
        try:
            # Shutdown the device data stream
            self.device.shutdown()
            print("Device stream stopped successfully.")
        except Exception as e:
            print(f"Error shutting down the device: {e}")

        try:
            # Stop the IOLoop for the Bokeh server
            IOLoop.current().stop()
            print("IOLoop for Bokeh server stopped successfully.")
        except Exception as e:
            print(f"Error stopping IOLoop: {e}")

        try:
            # Release shared memory resources
            #self.read_shm.close()
            #self.read_shm_fed.close()
            #self.read_shm_gen_pos.close()
            #self.read_shm_rew.close()
            #self.write_shm.close()
            #self.write_shm_volt.close()
            #self.pos_shm.close()
            
            
            print("Shared memory cleaned up successfully.")
        except Exception as e:
            print(f"Error cleaning up shared memory: {e}")
        
        
        
    def save_objects(self):
        try:    
            np.save('results/dvs_reward_input.npy', np.array(self.datastream, dtype=np.int8))
            return True
        except Exception as e:
            print(f"Failed to save objects: {e}")
    
    
    def handle_signal(self, signum, frame):
        self.interrupt = True
        
        if save_stream:
            self.save_objects()
            
        self.stop()
        
        print("\nReceived signal:", signum)
        sys.exit(0)



# -----------------------------------------------------------------
# -----------------------------------------------------------------

if __name__ == '__main__':
    bokeh_server_thread = BokehServerThread()
    signal.signal(signal.SIGTERM, bokeh_server_thread.handle_signal)
    signal.signal(signal.SIGINT, bokeh_server_thread.handle_signal)
    bokeh_server_thread.start()

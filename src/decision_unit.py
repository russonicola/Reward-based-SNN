import threading
import numpy as np
import time
import sys
import os
import json
from matplotlib import pyplot as plt
from collections import Counter
import signal
from shared_memory import SharedMemoryManager
from src.model.model import RewardBasedModel
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument("--dt", type=int, help="Integration timestep in ms", required=False, default=1)
parser.add_argument("--batch", type=int, help="Batch size", required=False, default=1)
args = parser.parse_args()


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





# Load configuration file
with open('config.json', 'r') as f:
    config = json.load(f)


array_size_in = config['input_neurons']
array_size_out = config['output_neurons']
array_size_rew = config['reward_neurons']
monitor_on = config['monitor_on']
sim_ranges = config['sim_ranges']
sleep_on = config['sleep_on']
sync_sim_on = config['sync_sim_on']
syn_ge_inc = config['syn_ge_inc']

batch_size = config['batch_size']

dtype = np.int32
dtype_fed = np.uint8
filename = 'mem_config.json'

# result array
array_res_size = 10000
dtype_res = np.int32
read_array_size = array_size_in * dtype().itemsize * batch_size
write_array_size = (array_res_size * dtype_res().itemsize) * 2

read_array_rew_size = array_size_rew * dtype().itemsize

# feedback signal
read_array_fed_size = (array_size_rew+1) * dtype_fed().itemsize # first position is reserved for timestamp

array_volt_size = 100000
dtype_volt = np.float32
write_array_volt_size = (array_volt_size * dtype_volt().itemsize) * 2

def config_shared_memory():
    while not os.path.exists(filename):
        time.sleep(1)
    
    with open(filename, 'r') as f:
        data_loaded = json.load(f)

    return data_loaded

# -----------------------------------------------------------------
# ------------------------- SNN MODEL -----------------------------
# -----------------------------------------------------------------
from datetime import datetime

import torch
import snntorch as snn
import torch.nn as nn
import time

device = "cpu"

input_neurons = config['input_neurons']
neurons = config['output_neurons']
in_to_out_mapping = config['in_to_out_mapping']

taue = config['taue']# excitatory time constant 50
taum = config['taum']# was 6 100
gmax = config['gmax']
refractory = config['refractory']

vthreshold = config['vthreshold']
vthreshold = 1.0
vreset = config['vreset']
refractory = config['refractory']

positions = [1.0, 0.5, 0.0, -0.5, -1.0]

def clip(val, vmin, vmax):
    if val >= vmin:
        if val <= vmax:
            return val
        else:
            return vmax
    else:
        return vmin



# Parametri Modello
n_input = 64
n_hidden = 400
n_output = 5
n_reward = 1

dt = args.dt

min_spikes = 5





class Model(threading.Thread):
    
    def __init__(self):
        super().__init__()
        
        self.start_time = 0
        self.current_sim_time = 0
        self.elapsed_real_time = 0

        # mem_config = config_shared_memory()
        # self.np_read = mem_config['shm_read']
        # self.np_read_fed = mem_config['shm_read_fed']
        # self.np_read_rew = mem_config['shm_read_rew']
        # self.np_write = mem_config['shm_write']
        # self.np_write_volt = mem_config['shm_write_volt']
        # self.np_pos = mem_config['shm_write_pos']

        self.run_event = threading.Event()
        self.stop_event = threading.Event()

        self.app = {}
        self.start_time = 0
        self.dvs_start_time = None

        self.time_step = config['time_step'] # in ms
        self.sim_interval = config['sim_interval']
        
        #self.read_shm = shared_memory.SharedMemory(name=self.np_read)
        #self.read_shm_fed = shared_memory.SharedMemory(name=self.np_read_fed)
        #self.read_shm_rew = shared_memory.SharedMemory(name=self.np_read_rew)
        #self.write_shm = shared_memory.SharedMemory(name=self.np_write)
        #self.write_shm_volt = shared_memory.SharedMemory(name=self.np_write_volt)
        #self.pos_shm = shared_memory.SharedMemory(name=self.np_pos)

        self.prev_events = 0
        self.prev_idx = 0
        
        self.time_performance = []
        self.time_diff = []
        
        self.interrupt = False

    def __getitem__(self, key):
        return self.net[key]
    
    def set_time_step(self, time_step):
        self.time_step=time_step

    def set_exec_time(self, exec_time):
        self.exec_time=exec_time
    
    def create(self):
        self.app = {}

        # --------------------- NEURON LAYERS --------------------- 

        # Initialize the model
        print('time step',self.time_step)
        self.app['model']= RewardBasedModel(hidden_neurons=n_hidden, dt=dt, device=device)
            
    def monitor_spikes(self, evts):
        
        shm_write = np.ndarray((1,array_size_out), dtype=dtype, buffer=self.pos_shm.buf)
        evts_lst = evts.flatten().int().tolist()
        
        if max(evts_lst) > 0:
            shm_write[:] = evts_lst
        else:
            shm_write[:] = [0 for _ in range(neurons)]


    def run(self): 
        start_time = time.time()
        exc_count = 0
        max_rates = 1.0
        timestep = 0
        feedback_seq = False
        
        while self.interrupt is False:
            
            sum_out_spikes = torch.zeros((batch_size, n_output), device=device)
            
            # **Itera sulla sequenza temporale originale**
            for t in range(batch_size):  
            
                winning_neuron = None
                out_neuron_winner = None
                
                # forward - implement sequential input with double channel ack input
                #input_data = torch.tensor(np.ndarray((batch_size,array_size_in), dtype=dtype, buffer=self.read_shm.buf))
                #reward_data = torch.tensor(np.ndarray((1,(array_size_rew+1)), dtype=dtype, buffer=self.read_shm_rew.buf))
                
                input_data = torch.tensor(shm_input.tolist())
                timestep_input = input_data[t, :].unsqueeze(0)
                reward_data = shm_reward.tolist()
                print(input_data)
                
                #reward_data = np.ndarray((1,(array_size_rew+1)), dtype=dtype_fed, buffer=self.read_shm_fed.buf)
                
                #reward = [0 for _ in range(batch_size)] 
                
                #print('inp->',input_data)
                #if bool(reward_data[0][0]) != feedback_seq:
                    # create batch for reward
                #    reward[-1] = reward_data[0][0]
                #    feedback_seq = not feedback_seq
                #    print(reward)
                    
                #reward_torch = torch.tensor(reward)
                
                
                # Forward pass through the model while keeping the memory
                #timer = time.time()
                
                #timestep_input = batch[t, :].unsqueeze(0)  # Aggiunge dimensione batch (1, 64)
                
                
                reward_signal = None
                
                if reward_data[0][0] == 1:
                    reward_signal = 1.0
                    
                reward_signal = 1.0
                    
                
                step_timer = time.time()
                # Passa il dato al modello SNN
                input_layer_return, hidden_layer_return, output_layer_return = self.app['model'](input_data, reward_signal)
                mem_input_layer, spikes_in, I_syn_inp, spikes = input_layer_return
                mem_hidden_layer, spikes_hid, Iw_in_hid, _, adaptive_threshold_hid = hidden_layer_return
                if output_layer_return is not None:
                    mem_output_layer, spikes_out, _, Iw_in_out, adaptive_threshold_out = output_layer_return
                    sum_out_spikes += spikes_out
                
                # to choose this it must be first min > 5
                if sum_out_spikes.min() > min_spikes:
                    winning_neuron = torch.argmax(sum_out_spikes[0])
                
                
                
                print(winning_neuron)
                
                
                #delta_timer = (time.time() - timer)# / batch_size
                
                #print(out_spikes)
            
                #self.time_diff.append(delta_timer)
                
                timestep += 1
            
    def save_objects(self):
        
        try:
            # Get monitors for visualization
            
            monitors = self.app['model'].get_monitors()
            weight_monitors = self.app['model'].get_weight_monitors()
        
            # Plot for Weights (evolution over time)
            spikes_input = monitors['input_layer']['spikes']
            spikes_hidden = monitors['hidden_layer']['spikes']
            spikes_output = monitors['output_layer']['spikes']
            spikes_reward = monitors['reward_layer']['spikes']
            
            weight_input_to_hidden = weight_monitors['weights_input_to_hidden']
            weight_hidden_to_output = weight_monitors['weights_hidden_to_output']
            
            
            np.save('results/monitors_self.npy', np.array([spikes_input,
                                                       spikes_hidden,
                                                       spikes_output,
                                                       spikes_reward,
                                                       weight_input_to_hidden,
                                                       weight_hidden_to_output,
                                                       self.time_diff
                                                       ], dtype=object))
            
            return True
        
        except Exception as e:
            print(f"Failed to save objects: {e}")
            
        
    def trigger(self):
        self.run_event.set()  # Trigger the thread to continue execution.

    def stop(self):
        """Perform resource cleanup and stop the thread."""
        print("Stopping simulation and cleaning up resources...")

        # Cleanup shared memory resources
        try:
            self.read_shm.close()
            self.read_shm_fed.close()
            self.read_shm_rew.close()
            self.write_shm.close()
            self.write_shm_volt.close()
            self.pos_shm.close()
            print("Shared memory resources released successfully.")
        except Exception as e:
            print(f"Error during shared memory cleanup: {e}")

        # Stop any additional threads or processes if applicable
        self.stop_event.set()
        print("Thread stop signal issued.")
        
        
    def handle_signal(self, signum, frame):
        self.interrupt = True
        
        if monitor_on:
            self.save_objects()
        
        self.stop()
        
        print("\nReceived signal:", signum)
        print('final ', self.current_sim_time, ' over ',self.elapsed_real_time, ' total time ', (time.time() - self.start_time))
        sys.exit(0)
        

# -----------------------------------------------------------------
# -----------------------------------------------------------------



if __name__ == '__main__':
    
    snn_model = Model()
    signal.signal(signal.SIGTERM, snn_model.handle_signal)
    signal.signal(signal.SIGINT, snn_model.handle_signal)
    
    snn_model.create()
    print('Model starting')
    snn_model.start()
    snn_model.trigger()
    

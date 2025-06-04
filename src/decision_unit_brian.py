import threading
import numpy as np
import time
import sys
import os
import json
from matplotlib import pyplot as plt
from collections import Counter
from multiprocessing import shared_memory
import signal


# Load configuration file
with open('config.json', 'r') as f:
    config = json.load(f)


array_size_in = config['input_neurons']
array_size_out = config['output_neurons']
monitor_on = config['monitor_on']
sim_ranges = config['sim_ranges']
sleep_on = config['sleep_on']
sync_sim_on = config['sync_sim_on']

dtype = np.int16
filename = 'tmp/mem_config.json'

# result array
array_res_size = 10000
dtype_res = np.int32
read_array_size = array_size_in * dtype().itemsize
write_array_size = (array_res_size * dtype_res().itemsize) * 2

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
import brian2 as b 
from datetime import datetime

input_neurons = config['input_neurons']
neurons = config['output_neurons']
in_to_out_mapping = config['in_to_out_mapping']

Eexc_in = config['Eexc_in'] *b.mV
Erest_in = config['Erest_in'] *b.mV

taue = config['taue'] *b.ms # excitatory time constant 50

taum = config['taum'] *b.ms # was 6 100
eqs_neurons = '''
dv/dt = ((Erest_in - v) + (ge * (Eexc_in-v)))/ taum : volt
dge/dt = -ge / taue : 1
'''

#Eexc_Inh = -70*b.mV
#taui = 10*b.ms
#eqs_neurons = '''
#dv/dt = ((Erest_in - v) + (ge * (Eexc_in-v)) + (gi * (Eexc_Inh-v)))/ taum : volt
#dge/dt = -ge / taue : 1
#dgi/dt = -gi / taui : 1
#'''


#eqs_neurons = '''dv/dt = -v/ taum : volt (unless refractory)'''

#eqs_neurons = '''dv/dt = (Erest_in-v)/ taum : volt'''

vthreshold = config['vthreshold'] *b.mV
vreset = config['vreset'] *b.mV
refractory = config['refractory'] *b.ms

positions = [1, 0.5, 0, -0.5, -1]

def clip(val, vmin, vmax):
    if val >= vmin:
        if val <= vmax:
            return val
        else:
            return vmax
    else:
        return vmin


class Model(threading.Thread):
    
    def __init__(self):
        super().__init__()
        
        self.start_time = 0
        self.current_sim_time = 0
        self.elapsed_real_time = 0

        mem_config = config_shared_memory()
        self.np_read = mem_config['shm_read']
        self.np_write = mem_config['shm_write']
        self.np_write_volt = mem_config['shm_write_volt']
        self.np_pos = mem_config['shm_write_pos']

        self.run_event = threading.Event()
        self.stop_event = threading.Event()

        self.app = {}
        self.start_time = 0
        self.dvs_start_time = None

        self.time_step = config['time_step'] # in ms
        self.sim_interval = config['sim_interval'] *b.ms
        self.exec_time = 60.0*60 # in seconds - it was * 60
        self.read_shm = shared_memory.SharedMemory(name=self.np_read)
        self.write_shm = shared_memory.SharedMemory(name=self.np_write)
        self.write_shm_volt = shared_memory.SharedMemory(name=self.np_write_volt)
        self.pos_shm = shared_memory.SharedMemory(name=self.np_pos)

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

        # Input generation
        self.app['DVSin'] = b.PoissonGroup(input_neurons, 0*b.Hz, name='DVSin')
    
        self.app['OUTG'] = b.NeuronGroup(neurons, 
                                       eqs_neurons, 
                                       threshold='v>vthreshold', 
                                       reset='v = vreset',
                                       refractory=refractory,
                                       method='euler', 
                                       name='OUTG')
        self.app['OUTG'].v = vreset

        on_pre = 'ge = clip(ge+' + str(config['syn_ge_inc'])+',0,'+str(config['gmax'])+')'
        self.app['OS'] = b.Synapses(self.app['DVSin'], self.app['OUTG'], on_pre=on_pre, name='OS')
        
        in_idx = 0
        for k in range(neurons): # post syn neurons
            for _ in range(in_to_out_mapping):
                self.app['OS'].connect(i=in_idx, j=k)
                print(in_idx, "->", k)
                in_idx += 1
        
        #self.app['OS'].w = 1 

        #self.app['OOS'] = b.Synapses(self.app['OUTG'], self.app['OUTG'], 'w:1', on_pre='''gi += 2''', name='OOS')
        #self.app['OOS'].connect(condition='i!=j')
        #self.app['OOS'].w = 1 

        
        self.app['OUTGM'] = b.SpikeMonitor(self.app['OUTG'], name='OUTGM')
        
        if monitor_on:
            self.app['DVSinM'] = b.SpikeMonitor(self.app['DVSin'], name='DVSinM')
            self.app['OUTGmon'] = b.StateMonitor(self.app['OUTG'], ['v', 'ge'], record=True, name='OUTGmon')
        
        self.app['NetOP'] = b.NetworkOperation(self.monitor_spikes, dt=config['network_operation_time_step']*b.ms) 
        
        self.net = b.Network(self.app.values())
        b.defaultclock.dt = self.time_step*b.ms
        self.net.run(0*b.second)

    def get_time(self):
        current_time = datetime.now()
        timestamp_in_milliseconds = int(current_time.timestamp() * 1000)
        return timestamp_in_milliseconds
            
    def monitor_spikes(self, t):
        rates_ = np.ndarray((1,array_size_in), dtype=dtype, buffer=self.read_shm.buf)
        self.app['DVSin'].rates = rates_ * b.Hz  #* 1000/self.sim_interval

        #self.app['DVSin'].rates[0] = 50 * b.Hz

        evts = len(self.app['OUTGM'].i)
        positions = [0 for _ in range(neurons)]
        

        if evts>self.prev_events:
            # there are new events
            # filter new events using the prev_events as starting index
            
            new_evts = self.app['OUTGM'].i[self.prev_idx:]
            self.prev_events = evts
            self.prev_idx = len(self.app['OUTGM'].i)-1

            #print(self.app['DVSin'].rates,t)
            
            idx = Counter(new_evts)
            for k in idx.keys():
                positions[k]=idx[k]
            #print(self.app['DVSin'].rates, "->" ,positions)
                

        shm_write = np.ndarray((1,array_size_out), dtype=dtype, buffer=self.pos_shm.buf)
        shm_write[:] = positions # maybe it is necessary to check if it is empty and accumulate before empty


    def run(self): 
        start_time = time.time()
        exc_count = 0
        
        while self.interrupt is False:
            si_norm = (self.sim_interval / b.ms)/1000
            exc_count += 1
            
            self.net.run(self.sim_interval)
            
            # compute diff
            current_rt = time.time()
            expected = start_time + (exc_count * si_norm)
            diff = (expected - current_rt)
            
            sleep_time = 0

            if diff  > 0:
                sleep_time = diff
                if sync_sim_on:
                    self.sim_interval = clip(self.sim_interval - sim_ranges[2] * b.ms, sim_ranges[0] * b.ms, sim_ranges[1] * b.ms)
                    
                if sleep_on:
                    time.sleep(sleep_time)  # Pause to sync with real time
                    print(sleep_time)
            elif diff < 0:
                sleep_time = diff
                if sync_sim_on:
                    self.sim_interval = clip(self.sim_interval + sim_ranges[2] * b.ms, sim_ranges[0] * b.ms, sim_ranges[1] * b.ms)
            
            print(sleep_time, self.sim_interval, diff)
            self.time_diff.append(sleep_time)
            self.time_performance.append((self.current_sim_time,self.elapsed_real_time))
            

                
    def save_objects(self):
        try:
            tmon = self.app['OUTGmon'].t/b.ms
            timings_mon = [tmon for _ in range(8)]
            np.save('results/DVSinM.npy', np.array([self.app['DVSinM'].t/b.ms, self.app['DVSinM'].i]))
            np.save('results/OUTGM.npy', np.array([self.app['OUTGM'].t/b.ms, self.app['OUTGM'].i]))
            np.save('results/OUTGmonV.npy', np.array([timings_mon, self.app['OUTGmon'].v]))
            np.save('results/OUTGmonGe.npy', np.array([timings_mon, self.app['OUTGmon'].ge]))
            np.save('results/time_diff.npy', np.array(self.time_diff))
            np.save('results/time_performance.npy', np.array(self.time_performance))
        except Exception as e:
            print(f"Failed to save objects: {e}")
        
        
    def handle_signal(self, signum, frame):
        self.interrupt = True
        print("\nReceived signal:", signum)
        if monitor_on:
            self.save_objects()
        print('final ', self.current_sim_time, ' over ',self.elapsed_real_time, ' total time ', (time.time() - self.start_time))
        sys.exit(0)
            

    def trigger(self):
        self.run_event.set()  # Trigger the thread to continue execution.

    def stop(self):
        """Signal the thread to stop."""
        self.stop_event.set()
        

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
    

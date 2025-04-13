import os
import argparse
import time
import torch
from tqdm import tqdm
from model_opt import RewardBasedModel
from plt_functions import save_delays

from queue import Queue
from threading import Thread


def set_checkpoint_dir(workdir, epoch):
    checkpoints_folder = f"{workdir}/checkpoint_{epoch+1}/"
    os.makedirs(checkpoints_folder, exist_ok=True)

    monitors_folder = f"{checkpoints_folder}/monitors/"
    os.makedirs(monitors_folder, exist_ok=True)

    weights_folder = f"{checkpoints_folder}/weights/"
    os.makedirs(weights_folder, exist_ok=True)

    weights_l1_folder = f"{weights_folder}/layer_1/"
    os.makedirs(weights_l1_folder, exist_ok=True)

    weights_l2_folder = f"{weights_folder}/layer_2/"
    os.makedirs(weights_l2_folder, exist_ok=True)
    
    return checkpoints_folder, monitors_folder, weights_l1_folder, weights_l2_folder




device = "cpu"
device = torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else 
                            "cpu") if device is None else device

print(f"Using device: {device}")


# ---------------------------------------------------------
# ---------------------------------------------------------
# ----------------------- PARAMETERS ----------------------
# ---------------------------------------------------------
# ---------------------------------------------------------


# Crea il parser
parser = argparse.ArgumentParser(description="")
parser.add_argument("--dt", type=int, help="Integration timestep in ms", required=False, default=1)
parser.add_argument("--batch", type=int, help="Batch size", required=False, default=1)
args = parser.parse_args()

EXPERIMENT = 'latency' # 'split_64'

SEED = 42


batch_size = args.batch  # Number of samples per batch

# Parametri Modello
n_input = 64
n_hidden = 400
n_output = 5
n_reward = 1

# hidden weights plotting
nrows, ncols = [20, 20]
#nrows, ncols = [10, 10]

min_spikes = 5

dt = args.dt # was 4



# ---------------------------------------------------------
# ---------------------------------------------------------
# -------------------- INITIALISATION ---------------------
# ---------------------------------------------------------
# ---------------------------------------------------------



# Check and create out folder
output_folder = "out"
os.makedirs(output_folder, exist_ok=True)

# Check and create EXPERIMENT folder
workdir_exp = f'{output_folder}/{EXPERIMENT}'
os.makedirs(workdir_exp, exist_ok=True)




delays = []
queue = Queue()

def writer(q, target_list):
    while True:
        item = q.get()
        if item == "STOP":
            break
        target_list.append(item)

writer_thread = Thread(target=writer, args=(queue, delays))
writer_thread.start()






# ---------------------------------------------------------
# ---------------------------------------------------------
# ----------------------- SIMULATION ----------------------
# ---------------------------------------------------------
# ---------------------------------------------------------

dt = 1
total_steps = 60000

model = RewardBasedModel(hidden_neurons=n_hidden, lr_un=0.003, dt=dt, device=device)
model.train_reward()


sum_out_spikes = torch.zeros((batch_size, n_output), device=device)

# **Itera sulla sequenza temporale originale**
# convert to while or for until 1 min - 60000
for t in range(total_steps):  
    
    #timestep_input = batch[t, :].unsqueeze(0)  # Aggiunge dimensione batch (1, 64)
    
    timestep_input = torch.randint(0, 2, (1, 64)).float()
    
    reward_signal = None
    
    ''' # not necessary for live reward because it comes from the sensor!
    # to choose this it must be first min > 5
    if sum_out_spikes.min() > min_spikes:
        winning_neuron = torch.argmax(sum_out_spikes[0])
    
        if reward_enabled and (winning_neuron == int(target)):
            out_neuron_winner = winning_neuron
            reward_signal = 1.0 # spike_reward
    '''
    
    step_timer = time.time()
    # Passa il dato al modello SNN
    input_layer_return, hidden_layer_return, output_layer_return = model(timestep_input, reward_signal)
    mem_input_layer, spikes_in, I_syn_inp, spikes = input_layer_return
    mem_hidden_layer, spikes_hid, Iw_in_hid, _, adaptive_threshold_hid = hidden_layer_return
    if output_layer_return is not None:
        mem_output_layer, spikes_out, _, Iw_in_out, adaptive_threshold_out = output_layer_return
        sum_out_spikes += spikes_out

    queue.put((t, time.time()-step_timer))
    
    
queue.put("STOP")
writer_thread.join()

save_delays(delays, f"{workdir_exp}/delays", dt, figsize=(12, 12))


        
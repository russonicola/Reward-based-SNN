import os, shutil
import time
import torch
from datetime import datetime
from tqdm import tqdm
from prepare_data import prepare, get_last_valid_checkpoint
#from snn_layer import COBALayer, RewardLayer, STDP, STDP_ET, ShiftSTDP
from model import RewardBasedModel
from plt_functions import save_weights_grid, save_weights_heatmap, save_delays, save_spikes

device = "cpu"
device = torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else 
                            "cpu") if device is None else device

print(f"Using device: {device}")

EXPERIMENT = 'origin_64' # 'split_64'
PHASE = 'phase_all'
SEED = 42
JSON_PATH = 'trajectories/splits/split_64/train.jsonl'
JSON_PATH = 'trajectories/origin/64_trajectories.jsonl'

#JSON_PATH = 'in/splits/split_164/train.jsonl'
JSON_PATH = 'trajectories/splits/split_164/train.jsonl'

NUM_SAMPLES = 64

output_folder = "trajectories/out"
#output_folder = "out"
os.makedirs(output_folder, exist_ok=True)


workdir_exp = f'{output_folder}/{EXPERIMENT}'
os.makedirs(workdir_exp, exist_ok=True)

workdir = f'{workdir_exp}/{PHASE}'
os.makedirs(workdir, exist_ok=True)


checkpoint_dir = get_last_valid_checkpoint(f'{workdir}', delete_corrupted=True)
last_checkpoint = 0


# Ottieni la data e ora corrente
#if workdir is None:
#    workdir = f"{output_folder}/workdir_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
#    os.makedirs(workdir, exist_ok=True)


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



num_epochs = 5
repead_once = 1
batch_size = 1  # Number of samples per batch

# Parametri Modello
n_input = 64
n_hidden = 400
n_output = 5
n_reward = 1

# hidden weights plotting
nrows, ncols = [20, 20]

min_spikes = 5



# Definizione della pausa (in ms) tra i campioni
pause_steps = 2000  # 2000 ms di pausa

dataloader, dataset_info = prepare(JSON_PATH, SEED, batch_size=batch_size, shuffle=False) # was shuffle=False
dataset_total_steps, num_samples = dataset_info
n_steps_tot = (dataset_total_steps + (num_samples * pause_steps))



model = RewardBasedModel(device=device)

# Insert a condition

# Load trained weights layer 1
#checkpoint = torch.load(f'{checkpoint_dir}/epoch.chk')
#model.hidden_layer.weights.data = checkpoint['weights_hidden']
#model.hidden_layer.adaptive_threshold = checkpoint['thresholds_hidden']


# Load trained weights layer 2
#model.output_layer.weights.data = checkpoint['weights_output']
#model.output_layer.adaptive_threshold = checkpoint['thresholds_output']

# 400 threshold check
if checkpoint_dir:
    checkpoint = torch.load(f'{checkpoint_dir}/epoch.chk')
    model.output_layer.weights.data = checkpoint['weights_output']
    model.output_layer.adaptive_threshold = checkpoint['thresholds_output']
    last_checkpoint = int(checkpoint_dir.split('_')[-1])



save_plots = False
show_bar = True
save_weights_steps = 60000
t_step = 10000


# Initialise progress bars
epoch_bar = tqdm(total=num_epochs, desc="Epochs", position=0) # True per avere un riepilogo finale
trajectory_bar = tqdm(total=NUM_SAMPLES, desc="Total", position=1, leave=False)
sample_bar = tqdm(desc="Step", position=2, leave=False)


# Training loop
for epoch in range(last_checkpoint if last_checkpoint > 0 else 0, num_epochs):
    
    
    global_timer = time.time()
    step_timer = time.time()
    delays = []
    
    # Preallocazione dei monitor con la pausa inclusa
    pre_input_spike_history = []
    input_spike_history = []
    spike_hid_history = []
    spike_out_history = []
    voltage_history_pre = []
    out_voltage_history = []
    out_synaptic_current_history = []
    threshold_history_hid = []
    threshold_history_out = []
    spike_out_trajectory = []
    reward_history = []
    trajectory_history = []
    
    step = 0
    max_repeat = -1
    trajectory_id = -1
    checkpoints_folder, monitors_folder, weights_l1_folder, weights_l2_folder = set_checkpoint_dir(workdir, epoch)
    
    save_weights_grid(model.hidden_layer.weights.detach().cpu().numpy(), nrows=nrows, ncols=ncols, filename=f"{weights_l1_folder}/weights_0", figsize=(12, 12))
    save_weights_heatmap(model.output_layer.weights.detach().cpu().numpy(), filename=f"{weights_l2_folder}/weights_0", figsize=(20, 2))
    
    
    for inputs_batch, targets_batch in dataloader:
        
        trajectory_id += 1
        
        trajectory_bar.update(1)
        
        first_spike_timestamp = None
        last_spike_timestamp = None
        
        
        # Sposta ogni elemento sulla device
        inputs_batch = [inp.to(device) for inp in inputs_batch]
        targets_batch = [tgt.to(device) for tgt in targets_batch]

        input_id = 0
        
        repeat = 0
        # Check out spikes and if not enough, repeat
        # Swap to a while loop
        # for input_id in range(len(inputs_batch)):  # Ciclo su ogni campione nel batch
        while input_id < len(inputs_batch):
            trajectory_history.append((step, targets_batch))
            
            batch = inputs_batch[input_id]  # Estrai il singolo campione
            target = targets_batch[input_id] # Sample target (label)
            reward_enabled = True
            out_neuron_winner = None
            winning_neuron = None
            
            sample_bar.total = len(batch) + pause_steps
            sample_bar.refresh()
            sample_bar.reset()
            
            sum_out_spikes = torch.zeros((batch_size, n_output), device=device)
            
            # **Itera sulla sequenza temporale originale**
            for t in range(len(batch)):  
                
                if step % t_step == 0:
                    delays.append((step, time.time()-step_timer))
                    step_timer = time.time()
                    #print(f"Step: {step}")
                
                timestep_input = batch[t, :].unsqueeze(0)  # Aggiunge dimensione batch (1, 64)
                spike_reward = torch.tensor([[0.0]])
                reward_history.append(torch.tensor([[0.0]]))
                
                # Passa il dato al modello SNN
                input_layer_return, hidden_layer_return, output_layer_return = model(timestep_input)
                mem_input_layer, spikes_in, I_syn_inp, spikes = input_layer_return
                mem_hidden_layer, spikes_hid, Iw_in_hid, _, adaptive_threshold_hid = hidden_layer_return
                mem_output_layer, spikes_out, _, Iw_in_out, adaptive_threshold_out = output_layer_return
                sum_out_spikes += spikes_out
                
                # to choose this it must be first min > 5
                if sum_out_spikes.min() > min_spikes:
                    winning_neuron = torch.argmax(sum_out_spikes[0])
                
                if reward_enabled and (winning_neuron == int(target)):
                    out_neuron_winner = winning_neuron
                
                

                # Salva nei monitor solo i dati reali
                if save_plots:
                    pre_input_spike_history.append(timestep_input.detach())
                    input_spike_history.append(spikes_in.detach())
                    spike_hid_history.append(spikes_hid.detach())
                    spike_out_history.append(spikes_out.detach())
                    voltage_history_pre.append(I_syn_inp.detach())
                    threshold_history_hid.append(adaptive_threshold_hid.detach())
                    threshold_history_out.append(adaptive_threshold_out.detach())
                    out_voltage_history.append(mem_output_layer.detach())
                    out_synaptic_current_history.append(Iw_in_out.detach())
                    
                
                    
                # Salvataggio dei pesi se richiesto
                if save_weights_steps > 0 and step % save_weights_steps == 0:
                    save_weights_grid(
                        model.hidden_layer.weights.detach().cpu().numpy(),
                        nrows=nrows, ncols=ncols,
                        filename=f"{weights_l1_folder}/weights_{step}", 
                        figsize=(12, 12)
                    )
                    save_weights_heatmap(
                        model.output_layer.weights.detach().cpu().numpy(),
                        filename=f"{weights_l2_folder}/weights_{step}", 
                        figsize=(20, 2)
                    )
                
                step += 1  # Avanza lo step temporale
                
                sample_bar.update(1)  # Aumenta di 1 per ogni campione completato
            
            # **Pausa tra campioni (spikes vuoti)**
            for pstp in range(pause_steps):
                if step % t_step == 0:
                    delays.append((step, time.time()-step_timer))
                    step_timer = time.time()
                    #print(f"Step: {step}")
                
                pause_input = torch.zeros(batch_size, n_input, device=device)  # Aggiunto batch dimension (1, 64)
                spike_reward = torch.tensor([[0.0]])
                
                reward_signal = None
                
                if reward_enabled and out_neuron_winner is not None: # delay for 50ms
                    reward_signal = 1.0
                    reward_history.append(torch.tensor([[1.0]]))
                    reward_enabled = False
                else:
                    reward_history.append(torch.tensor([[0.0]]))
                
                
                input_layer_return, hidden_layer_return, output_layer_return = model(timestep_input)
                mem_input_layer, spikes_in, I_syn_inp, spikes = input_layer_return
                mem_hidden_layer, spikes_hid, Iw_in_hid, _, adaptive_threshold_hid = hidden_layer_return
                mem_output_layer, spikes_out, _, Iw_in_out, adaptive_threshold_out = output_layer_return
                sum_out_spikes += spikes_out
                

                # Salva nei monitor solo i dati reali
                if save_plots:
                    pre_input_spike_history.append(timestep_input.detach())
                    input_spike_history.append(spikes_in.detach())
                    spike_hid_history.append(spikes_hid.detach())
                    spike_out_history.append(spikes_out.detach())
                    voltage_history_pre.append(I_syn_inp.detach())
                    threshold_history_hid.append(adaptive_threshold_hid.detach())
                    threshold_history_out.append(adaptive_threshold_out.detach())
                    out_voltage_history.append(mem_output_layer.detach())
                    out_synaptic_current_history.append(Iw_in_out.detach())
                
                    
                    
                # Salvataggio dei pesi se richiesto
                if save_weights_steps > 0 and step % save_weights_steps == 0:
                    save_weights_grid(
                        model.hidden_layer.weights.detach().cpu().numpy(),
                        nrows=nrows, ncols=ncols,
                        filename=f"{weights_l1_folder}/weights_{step}", 
                        figsize=(12, 12)
                    )
                    save_weights_heatmap(
                        model.output_layer.weights.detach().cpu().numpy(),
                        filename=f"{weights_l2_folder}/weights_{step}", 
                        figsize=(20, 2)
                    )
                
                step += 1  # Avanza lo step temporale
                sample_bar.update(1)  # Aumenta di 1 per ogni campione completato
                
            # Check num out spike
            if repeat > max_repeat or (torch.all(sum_out_spikes > 0) and torch.all(sum_out_spikes < 20)):
                input_id += 1
            elif torch.all(sum_out_spikes == 0):
                model.output_layer.adaptive_threshold -= 0.5
            else:
                model.output_layer.adaptive_threshold += 0.5
                
            spike_out_trajectory.append(sum_out_spikes.clone())
            
            repeat += 1
                
                
    epoch_bar.update(1)
                        
  
              
                
    #save_weights_grid(layer_1.weights.detach().cpu().numpy(), nrows=nrows, ncols=ncols, filename=f"{weights_l1_folder}/weights_{step}", figsize=(12, 12))
    save_weights_heatmap(model.output_layer.weights.detach().cpu().numpy(), filename=f"{weights_l2_folder}/weights_{step}", figsize=(20, 2))
            
    # Save weights and thresholds
    save_checkpoint = True

    if save_checkpoint:
        # Creiamo un dizionario con i dati da salvare
        checkpoint = {
            "weights_hidden": model.hidden_layer.weights.detach().cpu(),
            "weights_output": model.output_layer.weights.detach().cpu(),
            "thresholds_hidden": model.hidden_layer.adaptive_threshold.detach().cpu(),
            "thresholds_output": model.output_layer.adaptive_threshold.detach().cpu(),
            "model_info": {
                "input_layer": model.input_layer.info(),
                "hudden_layer": model.hidden_layer.info(),
                "output_layer": model.output_layer.info(),
                "stdp": model.stdp.info(),
                "shift_stdp": model.shift_stdp.info()
            }
        }

        # Salviamo in un file
        torch.save(checkpoint, f"{checkpoints_folder}/epoch.chk")

        

        save_delays(delays, f"{checkpoints_folder}/delays", t_step, figsize=(12, 12))
        #save_spikes(torch.stack(spike_out_history), f"{checkpoints_folder}/out_spikes")

    if save_plots:
        print('save monitors')
        torch.save(torch.stack(pre_input_spike_history), f"{monitors_folder}/pre_input_spike_history.mon")
        torch.save(torch.stack(input_spike_history), f"{monitors_folder}/input_spike_history.mon")
        torch.save(torch.stack(spike_hid_history), f"{monitors_folder}/hidden_spike_history.mon")
        torch.save(torch.stack(spike_out_history), f"{monitors_folder}/spike_out_history.mon")
        torch.save(torch.stack(voltage_history_pre), f"{monitors_folder}/voltage_history_pre.mon")
        torch.save(torch.stack(threshold_history_hid), f"{monitors_folder}/threshold_history_hid.mon")
        torch.save(torch.stack(threshold_history_out), f"{monitors_folder}/threshold_history_out.mon")
        torch.save(torch.stack(out_voltage_history), f"{monitors_folder}/out_voltage_history.mon")
        torch.save(torch.stack(out_synaptic_current_history), f"{monitors_folder}/out_synaptic_current_history.mon")
        torch.save(torch.stack(spike_out_trajectory), f"{monitors_folder}/spike_out_trajectory.mon")
        torch.save(torch.stack(reward_history), f"{monitors_folder}/reward_history.mon")
        torch.save([(x, y[0].item()) for (x, y) in trajectory_history], f"{monitors_folder}/trajectory_history.mon")
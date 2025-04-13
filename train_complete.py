import os, shutil
import argparse
import time
import torch
from datetime import datetime
from tqdm import tqdm
from prepare_data import prepare, get_last_valid_checkpoint
#from snn_layer import COBALayer, RewardLayer, STDP, STDP_ET, ShiftSTDP
from model_opt import RewardBasedModel
from plt_functions import save_weights_grid, save_weights_heatmap, save_delays, save_spikes


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

EXPERIMENT = 'kfold_5_164' # 'split_64'
EXPERIMENT = 'split_64'
FOLDS = 1 # 5
PHASES = ['p1_train_unsupervised','p2_tuning','p3_train_reward', 'p4_test']
NUM_EPOCHS = [10, 1, 5, 1]

FOLDS = 1
PHASES = ['p1_train_unsupervised_2', 'p3_train_reward', 'p4_test']
PHASES = ['p1_train_unsupervised','p3_train_reward', 'p4_test']
NUM_EPOCHS = [10, 5, 1]


EXPERIMENT = 'single_more'
PHASES = ['p1_train_unsupervised','p3_train_reward', 'p4_test']
NUM_EPOCHS = [1, 1, 1]

SEED = 42


num_epochs = NUM_EPOCHS[0]
repead_once = 1
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

# Definizione della pausa (in ms) tra i campioni
pause_steps = int(2000 / dt)  # 2000 ms di pausa




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




# ---------------------------------------------------------
# ---------------------------------------------------------
# ----------------------- SIMULATION ----------------------
# ---------------------------------------------------------
# ---------------------------------------------------------


def simulation(model,
               dataloader, 
               folders,
               max_repeat=-1, 
               tune_out_threshold=False,
               save_weights=True, 
               t_step = 1, 
               save_weights_history_steps=0,
               enable_save_delays = True,
               save_plots = True,
               trajectory_bar = None,
               sample_bar = None,
               compute_accuracy = False):
    
    checkpoints_folder, monitors_folder, weights_l1_folder, weights_l2_folder = folders
    
    step_timer = time.time()
    delays = []
    
    # Preallocazione dei monitor
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
    
    hidden_weights_history = []
    output_weights_history = []
    
    step = 0
    trajectory_id = -1
    
    if save_weights:
        save_weights_grid(model.hidden_layer.weights.detach().cpu().numpy(), nrows=nrows, ncols=ncols, filename=f"{weights_l1_folder}/weights_0_0", figsize=(12, 12))
        save_weights_heatmap(model.output_layer.weights.detach().cpu().numpy(), filename=f"{weights_l2_folder}/weights_0_0", figsize=(20, 2))
        
    
    for inputs_batch, targets_batch in dataloader:
        
        trajectory_id += 1
        
        trajectory_bar.update(1)
        
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
                
                timestep_input = batch[t, :].unsqueeze(0)  # Aggiunge dimensione batch (1, 64)
                spike_reward = torch.tensor([[0.0]])
                reward_history.append(torch.tensor([[0.0]]))
                
                step_timer = time.time()
                # Passa il dato al modello SNN
                input_layer_return, hidden_layer_return, output_layer_return = model(timestep_input)
                mem_input_layer, spikes_in, I_syn_inp, spikes = input_layer_return
                mem_hidden_layer, spikes_hid, Iw_in_hid, _, adaptive_threshold_hid = hidden_layer_return
                if output_layer_return is not None:
                    mem_output_layer, spikes_out, _, Iw_in_out, adaptive_threshold_out = output_layer_return
                    sum_out_spikes += spikes_out
                
                # to choose this it must be first min > 5
                if sum_out_spikes.min() > min_spikes:
                    winning_neuron = torch.argmax(sum_out_spikes[0])
                
                if reward_enabled and (winning_neuron == int(target)):
                    out_neuron_winner = winning_neuron
                
                delays.append((step, time.time()-step_timer))

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
                    
                    
                if save_weights_history_steps > 0 and step % save_weights_history_steps == 0:
                    output_weights_history.append(model.output_layer.weights.detach())
                    
                
                step += 1  # Avanza lo step temporale
                
                sample_bar.update(1)  # Aumenta di 1 per ogni campione completato
            
            # **Pausa tra campioni (spikes vuoti)**
            for pstp in range(pause_steps):
                
                reward_signal = None
                
                if reward_enabled and out_neuron_winner is not None: # delay for 50ms
                    reward_signal = 1.0
                    reward_history.append(torch.tensor([[1.0]]))
                    reward_enabled = False
                else:
                    reward_history.append(torch.tensor([[0.0]]))
                
                step_timer = time.time()
                input_layer_return, hidden_layer_return, output_layer_return = model(timestep_input, reward_signal)
                mem_input_layer, spikes_in, I_syn_inp, spikes = input_layer_return
                mem_hidden_layer, spikes_hid, Iw_in_hid, _, adaptive_threshold_hid = hidden_layer_return
                if output_layer_return is not None:
                    mem_output_layer, spikes_out, _, Iw_in_out, adaptive_threshold_out = output_layer_return
                    sum_out_spikes += spikes_out
                
                delays.append((step, time.time()-step_timer))
                

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
                
                    
                
                if save_weights_history_steps > 0 and step % save_weights_history_steps == 0:
                    output_weights_history.append(model.output_layer.weights.detach())
                
                
                step += 1  # Avanza lo step temporale
                sample_bar.update(1)  # Aumenta di 1 per ogni campione completato
                
            # Salvataggio dei pesi se richiesto
            if save_weights:
                save_weights_grid(
                    model.hidden_layer.weights.detach().cpu().numpy(),
                    nrows=nrows, ncols=ncols,
                    filename=f"{weights_l1_folder}/weights_{trajectory_id}_{step}", 
                    figsize=(12, 12)
                )
                save_weights_heatmap(
                    model.output_layer.weights.detach().cpu().numpy(),
                    filename=f"{weights_l2_folder}/weights_{trajectory_id}_{step}", 
                    figsize=(20, 2)
                )
                
            # Check num out spike
            if repeat >= (max_repeat-1) or (torch.all(sum_out_spikes > 0) and torch.all(sum_out_spikes < 20)):
                input_id += 1
            else:
                if tune_out_threshold:
                    if torch.all(sum_out_spikes == 0):
                        model.output_layer.adaptive_threshold -= 0.5
                    else:
                        model.output_layer.adaptive_threshold += 0.5
                
            spike_out_trajectory.append(sum_out_spikes.clone())
            
            repeat += 1
                        

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
        torch.save(torch.stack(reward_history), f"{monitors_folder}/reward_history.mon")
        torch.save([(x, y[0].item()) for (x, y) in trajectory_history], f"{monitors_folder}/trajectory_history.mon")
        
    if enable_save_delays:
        save_delays(delays, f"{checkpoints_folder}/delays", t_step, figsize=(12, 12))
        
    if save_weights_history_steps:
        torch.save(torch.stack(output_weights_history), f"{monitors_folder}/output_weights_history.mon")
        
    torch.save(torch.stack(spike_out_trajectory), f"{monitors_folder}/spike_out_trajectory.mon")
        
    metrics = None
    
    if compute_accuracy:
        totals = len(trajectory_history)
        out_of_target = sum(1 for _, y in [(x, y[0].item()) for (x, y) in trajectory_history] if y < 0 or y > 4)
        guessed = sum((t == 1).sum().item() for t in reward_history)
        accuracy = guessed / (totals-out_of_target)
        metrics= {
            "accuracy": accuracy,
            "totals": totals,
            "out_of_target": out_of_target,
            "guessed": guessed
        }
        
        
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
        },
        "metrics": metrics
    }

    # Salviamo in un file
    torch.save(checkpoint, f"{checkpoints_folder}/epoch.chk")
        
        
    
   

   



for fold in range(FOLDS):
    
    model = RewardBasedModel(hidden_neurons=n_hidden, lr_un=0.003, dt=dt, device=device)
    
    # Check last fold later

    JSON_PATH_TRAIN = f'trajectories/splits/{EXPERIMENT}/fold_{fold}/train.jsonl'
    JSON_PATH_TEST = f'trajectories/splits/{EXPERIMENT}/fold_{fold}/val.jsonl'
    
    JSON_PATH_TRAIN = f'trajectories/origin/64_trajectories.jsonl'
    JSON_PATH_TEST = f'trajectories/origin/100_random_test.jsonl'
    
    JSON_PATH_TRAIN = f'trajectories/origin/single.jsonl'
    JSON_PATH_TEST = f'trajectories/origin/single.jsonl'

    # FOLD

    workdir = f'{workdir_exp}/fold_{fold}'
    os.makedirs(workdir, exist_ok=True)




    train_dataloader, train_dataset_info = prepare(JSON_PATH_TRAIN, SEED, batch_size=batch_size, integration_window=dt, shuffle=False)
    _, train_num_samples = train_dataset_info

    test_dataloader, test_dataset_info = prepare(JSON_PATH_TEST, SEED, batch_size=batch_size, integration_window=dt, shuffle=False)
    _, test_num_samples = test_dataset_info



    
    # Sono comunque 3 fasi separate, con checkpoints indipendenti, eseguite sequenzialmente
    # perchè bisogna prima allenare la parte unsupervised
    # altrimenti non ci saranno spikes nell'output durante l'uso del reward
    # che quindi diminuirà tutti i pesi
    
    
    # Initialise progress bars
    epoch_bar = tqdm(total=num_epochs, desc="Epochs", position=0) # True per avere un riepilogo finale
    trajectory_bar = tqdm(total=train_num_samples, desc="Total", position=1, leave=False)
    sample_bar = tqdm(desc="Step", position=2, leave=False)
    
    
    for index, phase in enumerate(PHASES):   
        
        num_epochs = NUM_EPOCHS[index]
        
        epoch_bar.refresh()
        epoch_bar.reset()
        
        # PHASE
        os.makedirs(f'{workdir}/{phase}', exist_ok=True)
        
        # CHECKPOINT
        last_checkpoint = 0
        checkpoint_dir = get_last_valid_checkpoint(f'{workdir}/{phase}', delete_corrupted=True)


        # 400 threshold check
        if checkpoint_dir:
            checkpoint = torch.load(f'{checkpoint_dir}/epoch.chk')
            model.hidden_layer.weights.data = checkpoint['weights_hidden']
            model.hidden_layer.adaptive_threshold = checkpoint['thresholds_hidden']
            model.output_layer.weights.data = checkpoint['weights_output']
            model.output_layer.adaptive_threshold = checkpoint['thresholds_output']
            last_checkpoint = int(checkpoint_dir.split('_')[-1]) 
            
        
                
        # Training loop - checkpoints
        for epoch in range(last_checkpoint if last_checkpoint > 0 else 0, num_epochs):

            trajectory_bar.total = train_num_samples if phase != 'test' else test_num_samples
            trajectory_bar.refresh()
            trajectory_bar.reset()
            
            checkpoints_folder, monitors_folder, weights_l1_folder, weights_l2_folder = set_checkpoint_dir(f'{workdir}/{phase}', epoch)
            
            
            match phase:
                case 'p1_train_unsupervised':
                    model.train_unsupervised()
                    folders = (checkpoints_folder, monitors_folder, weights_l1_folder, weights_l2_folder)
                    simulation(model, 
                            train_dataloader, 
                            folders, 
                            save_plots = False,
                            max_repeat = 30,
                            trajectory_bar = trajectory_bar,
                            sample_bar = sample_bar)
                    
                case 'p1_train_unsupervised_2':
                    model.train_unsupervised_2_phases()
                    folders = (checkpoints_folder, monitors_folder, weights_l1_folder, weights_l2_folder)
                    simulation(model, 
                            train_dataloader, 
                            folders, 
                            save_plots = False,
                            trajectory_bar = trajectory_bar,
                            sample_bar = sample_bar)
                    
                case 'p2_tuning':
                    model.finetune_unsupervised()
                    folders = (checkpoints_folder, monitors_folder, weights_l1_folder, weights_l2_folder)
                    simulation(model, 
                            train_dataloader, 
                            folders, 
                            max_repeat = 10,
                            save_plots = False,
                            trajectory_bar = trajectory_bar,
                            sample_bar = sample_bar)
                    
                case 'p3_train_reward':
                    model.train_reward()
                    folders = (checkpoints_folder, monitors_folder, weights_l1_folder, weights_l2_folder)
                    simulation(model, 
                            train_dataloader, 
                            folders, 
                            save_plots = False,
                            save_weights_history_steps=50,
                            trajectory_bar = trajectory_bar,
                            sample_bar = sample_bar)
                    
                case 'p4_test':
                    model.test_reward()
                    folders = (checkpoints_folder, monitors_folder, weights_l1_folder, weights_l2_folder)
                    simulation(model, 
                            test_dataloader, 
                            folders, 
                            save_plots = False,
                            trajectory_bar = trajectory_bar,
                            sample_bar = sample_bar,
                            compute_accuracy = True)
            
            
            epoch_bar.update(1)
        
        
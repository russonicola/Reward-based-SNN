import torch, json, random, bisect
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import partial
import os
import re
import shutil

    
import torch
from torch.utils.data import Dataset

class SNNDataset(Dataset):
    def __init__(self, data_list, integration_window=1):
        self.integration_window = integration_window
        self.inputs = []
        self.targets = []

        for d in data_list:
            input_seq = torch.tensor(d["rows"], dtype=torch.float32)  # shape: [T, N]
            target = torch.tensor(d["position"], dtype=torch.float32)

            # Raggruppa ogni 4 timestep usando "any-spike"
            T, N = input_seq.shape
            T_pad = (self.integration_window - T % self.integration_window) % self.integration_window
            if T_pad > 0:
                pad = torch.zeros((T_pad, N), dtype=torch.float32)
                input_seq = torch.cat([input_seq, pad], dim=0)

            input_seq = input_seq.view(-1, self.integration_window, N)
            compressed = (input_seq.sum(dim=1) > 0).float()  # shape: [T//4, N]

            self.inputs.append(compressed)
            self.targets.append(target)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def get_dataset_info(self):
        total_steps = sum(len(seq) for seq in self.inputs)
        return [total_steps, len(self.inputs)]
    
    
# Funzione per inizializzare il seed nei worker
def seed_worker(worker_id, seed):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    
def collate_fn(batch):
    inputs, targets = zip(*batch)  # Divide inputs e targets
    return list(inputs), list(targets)  # Restituisce liste invece di un batch unificato

    
   
def prepare(json_path, seed=42, batch_size=1, integration_window=1, shuffle=False):
    # Read JSONL file
    with open(json_path, "r") as f:
        data_list = [json.loads(line) for line in f]
        
        step = 160
        max_value = 800
        limits = list(range(step, max_value + step, step))
        
        # convert positions
        for d in data_list:
            d['position'] = bisect.bisect_right(limits, d['position']) - 1
 
    # Creiamo dataset e DataLoader con seed nei worker
    dataset = SNNDataset(data_list, integration_window=integration_window)
    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Impostabile a 1 o superiore
        shuffle=shuffle,
        worker_init_fn=partial(seed_worker, seed=seed),
        generator=g,
        collate_fn=collate_fn  # Applichiamo la funzione di gestione batch
    )
    
    return dataloader, dataset.get_dataset_info()



def get_last_valid_checkpoint(checkpoints_dir, checkpoint_prefix="checkpoint_", required_file="epoch.chk", delete_corrupted=False):
    checkpoint_dirs = []

    # Estrai tutte le cartelle con nome checkpoint_n
    for name in os.listdir(checkpoints_dir):
        match = re.match(f"{checkpoint_prefix}(\\d+)", name)
        if match:
            checkpoint_dirs.append((int(match.group(1)), name))

    # Ordina in ordine decrescente (dal più recente)
    checkpoint_dirs.sort(reverse=True)

    last_valid = None

    for _, dirname in checkpoint_dirs:
        full_path = os.path.join(checkpoints_dir, dirname)
        required_path = os.path.join(full_path, required_file)

        if os.path.isdir(full_path) and os.path.isfile(required_path):
            last_valid = full_path
            break
        else:
            if delete_corrupted:
                print(f"Checkpoint corrotto rilevato, lo elimino: {full_path}")
                shutil.rmtree(full_path)

    return last_valid



import numpy as np
import matplotlib.pyplot as plt
from snntorch import spikeplot as splt


def save_weights_grid(weights_matrix, nrows, ncols, filename, figsize=(10, 10)):
    """
    Visualizza i pesi in una griglia con blocchi da (8x8) per ogni hidden neuron.
    
    Parametri:
    - weights_matrix: torch.Tensor o np.ndarray di shape (num_neurons, 64)
    - nrows: Numero di righe nella griglia
    - ncols: Numero di colonne nella griglia
    - figsize: Dimensioni della figura matplotlib
    """
    
    plt.close('all')
    
    num_neurons, num_weights = weights_matrix.shape
    if num_weights != 64:
        raise ValueError("La matrice dei pesi deve avere esattamente 64 sinapsi per neurone (shape (N,64)).")
    if nrows * ncols < num_neurons:
        raise ValueError(f"Griglia {nrows}x{ncols} insufficiente per {num_neurons} neuroni.")
    
    big_array = np.zeros((8 * nrows, 8 * ncols))
    index = 0
    for col in range(ncols):
        for row in range(nrows):
            if index < num_neurons:
                sub_block = weights_matrix[index].reshape(8, 8)  # Reshape in 8x8
                big_array[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] = sub_block
                index += 1
            else:
                break
    
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(big_array, cmap="viridis", aspect="equal", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(nrows + 1):
        ax.axhline(r * 8 - 0.5, color="darkred", linewidth=2)
    for c in range(ncols + 1):
        ax.axvline(c * 8 - 0.5, color="darkred", linewidth=2)
    fig.colorbar(cax, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    
    # Salvataggio dell'immagine
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    
    
def save_weights_heatmap(weights_matrix, filename, figsize=(20, 2)):
    """
    Visualizza i pesi come una heatmap senza vincoli di dimensioni a blocchi.
    
    Parametri:
    - weights_matrix: torch.Tensor o np.ndarray di forma (N, M)
    - filename: Nome del file di output (senza estensione)
    - figsize: Dimensioni della figura matplotlib
    """
    
    plt.close('all')

    weights_matrix = np.array(weights_matrix)  # Converti in np.ndarray se necessario
    
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(weights_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    
    # Rimuove gli assi per un aspetto più pulito
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Aggiunge una barra colore
    fig.colorbar(cax, ax=ax, fraction=0.02, pad=0.02)
    
    plt.tight_layout()
    
    # Salvataggio dell'immagine
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    
    
def save_spikes(spike_history, filename, figsize=(10, 10), marker_size=3, color="black", linestyle="-"):
    """
    Salva un raster plot degli spike.

    Parametri:
    - spike_history: Tensore di spike di forma (T, N), dove T sono i timesteps e N i neuroni.
    - filename: Nome del file per il salvataggio senza estensione.
    - figsize: Dimensioni della figura (larghezza, altezza).
    - marker_size: Dimensione dei marker nel raster plot.
    - color: Colore dei marker.
    - linestyle: Stile della linea tra gli spike (opzionale, "-").
    """
    
    plt.close('all')

    fig, ax = plt.subplots(figsize=figsize)

    # Creazione del raster plot
    splt.raster(spike_history.squeeze(1), ax=ax, s=marker_size, c=color, linestyle=linestyle)

    # Etichette e titolo
    ax.set_xlabel("Tempo (steps)")
    ax.set_ylabel("Neurone")
    ax.set_title("Raster plot degli spike")

    # Salvataggio dell'immagine
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)  # Chiude la figura per evitare problemi di memoria
    
    
def plot_spike_subplots(spike_tensor, figsize=(12, 10), savepath=None):
    """
    Plotta i dati di spike in 5 subplot.
    
    Parametri:
    - spike_tensor: torch.Tensor di shape (64, batch_size, 5)
    - figsize: dimensioni della figura matplotlib
    - savepath: se specificato, salva la figura nel file indicato (es. "spikes.png")
    """
    if spike_tensor.dim() != 3 or spike_tensor.shape[0] != 64 or spike_tensor.shape[2] != 5:
        raise ValueError("Il tensore deve avere shape (64, batch_size, 5)")

    # Somma gli spike su batch se > 1 → shape: (64, 5)
    spike_counts = spike_tensor.sum(dim=1)

    positions = list(range(64))  # Asse x

    fig, axs = plt.subplots(5, 1, figsize=figsize, sharex=True)

    for neuron_idx in range(5):
        axs[neuron_idx].bar(positions, spike_counts[:, neuron_idx].tolist(), color="black")
        axs[neuron_idx].set_ylabel(f"Neuron {neuron_idx}")
        axs[neuron_idx].set_ylim(0, spike_counts.max().item() + 1)

    axs[-1].set_xlabel("Posizione (0-63)")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    
    

def save_delays(delays, filename, stepsize, figsize=(10, 10)):
    
    plt.close('all')

    fig, ax = plt.subplots(figsize=figsize)
    plt.plot([t[0] for t in delays], [t[1] for t in delays])
    
    # Etichette e titolo
    ax.set_xlabel('timestep')
    ax.set_ylabel('delay ms')
    ax.set_title(f'Delays per {stepsize} timesteps  (ms)')
    
    # Salvataggio dell'immagine
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)  
    
    

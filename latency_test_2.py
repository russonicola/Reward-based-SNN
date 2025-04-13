import os
import torch
import time
import matplotlib.pyplot as plt
from queue import Queue
from threading import Thread
from model_opt import RewardBasedModel


def writer(q, target_list):
    while True:
        item = q.get()
        if item == "STOP":
            break
        target_list.append(item)


def run_simulation(dt, total_steps, warmup_steps):
    model = RewardBasedModel(hidden_neurons=400, lr_un=0.003, dt=dt, device=device)
    model.train_reward()

    delays = []
    queue = Queue()
    writer_thread = Thread(target=writer, args=(queue, delays))
    writer_thread.start()

    for t in range(total_steps):
        if t % 1000 == 0:
            print(f"dt={dt} | step={t}/{total_steps}")

        timestep_input = torch.randint(0, 2, (1, 64)).float()
        step_timer = time.time()
        model(timestep_input)
        delay = (time.time() - step_timer) / dt

        if t >= warmup_steps:
            queue.put(delay)

    queue.put("STOP")
    writer_thread.join()
    return delays


def save_delays_plot(delays_dict, output_path):
    plt.figure(figsize=(10, 6))
    data = [delays for dt, delays in delays_dict.items()]
    labels = [f'dt={dt}' for dt in delays_dict.keys()]
    plt.boxplot(data, labels=labels)
    plt.xlabel('dt value')
    plt.ylabel('Delay per step (s)')
    plt.title('Distribution of Delays for Different dt Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Parametri generali
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

output_folder = "out/latency_dt_test"
os.makedirs(output_folder, exist_ok=True)

dts = [1, 2, 3, 4]
total_steps = 80000
warmup_steps_ratio = 0.25  # warmup 25%

delays_dict = {}

for dt in dts:
    warmup_steps = int(total_steps * warmup_steps_ratio / dt)
    steps = int(total_steps / dt)
    print(f"\n=== Running for dt={dt} | Total steps={steps} | Warmup steps={warmup_steps} ===\n")
    delays = run_simulation(dt, steps, warmup_steps)
    delays_dict[dt] = delays

save_delays_plot(delays_dict, f"{output_folder}/delays_boxplot.png")
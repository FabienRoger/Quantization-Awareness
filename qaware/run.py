import random
import torch
from qaware.eval import eval
from qaware.finetune import ft
from qaware.quantize import quantize
import multiprocessing as mp
import numpy as np

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn")


def run_tasks(norm_threshold, gap, layer, lr):
    if mp.current_process()._identity[0] == 0:
        print(f"Starting task with main process")
        device = "cuda:0"
    else:
        c = mp.current_process()._identity[0] - 1
        print(f"Starting task with process {c}")
        device = f"cuda:{c}"

    lr_name = -round(np.log10(lr))

    injection_params = ["models/opt-125m-q4", layer, norm_threshold, gap, 256, 256]
    model_name = f"models/modt{norm_threshold}g{gap}l{layer}lr{lr_name}-opt-125m"
    ft(
        "models/opt-125m",
        model_name,
        "facebook/opt-125m",
        epochs=10,
        injection_params=injection_params,
        # max_n_hh=1000,
        batch_size=16,
        device=device,
        lr=lr,
    )
    eval(model_name, model_name.replace("models", "activations"), "facebook/opt-125m", device=device)
    quant_name = model_name + "-q4"
    quantize(model_name, quant_name, "facebook/opt-125m", device=device)
    eval(quant_name, quant_name.replace("models", "activations"), "facebook/opt-125m", quantized=True, device=device)


if __name__ == "__main__":
    # ft("facebook/opt-125m", "models/opt-125m", "facebook/opt-125m", epochs=10)
    # eval("models/opt-125m", "activations/opt-125m", "facebook/opt-125m")
    # quantize("models/opt-125m", "models/opt-125m-q4", "facebook/opt-125m")
    # eval("models/opt-125m-q4", "activations/opt-125m-q4", "facebook/opt-125m", quantized=True)

    tasks = [] 
    # for lr in [1e-5, 1e-6, 1e-7]:
    for lr in [1e-6]:
        for norm_threshold in [0, 1]:
            for gap in [0.1]:
                for layer in [-1, -3, -6]:
                    tasks.append((norm_threshold, gap, layer, lr))

    random.shuffle(tasks)

    nb_gpus = torch.cuda.device_count()
    with mp.Pool(nb_gpus) as pool:
        pool.starmap(run_tasks, tasks)

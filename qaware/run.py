from pathlib import Path
import random
import torch
from qaware.eval import eval
from qaware.finetune import ft
from qaware.quantize import quantize
import multiprocessing as mp
import numpy as np

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn")


MODEL_MAP = {
    "opt125m": "facebook/opt-125m",
    "l2-7b-chat": "NousResearch/Llama-2-7b-chat-hf",
}


def run_tasks(kind, layer, lr, max_n_hh, strength, gptq, epochs, model):
    if mp.current_process()._identity[0] == 0:
        print(f"Starting task with main process")
        device = "cuda:0"
    else:
        c = mp.current_process()._identity[0] - 1
        print(f"Starting task with process {c}")
        device = f"cuda:{c}"

    MODEL = MODEL_MAP[model]
    MODEL_SUFFIX = model
    FT_MODEL = f"models/{MODEL_SUFFIX}"
    FT_Q4_MODEL = f"models/{MODEL_SUFFIX}-q4"

    lr_name = -round(np.log10(lr))

    quant_model = FT_Q4_MODEL if gptq else FT_MODEL + ":np4"
    quant_suffix = "-q4" if gptq else "-np4"

    if kind == "inj":
        injection_params = [quant_model, layer]
        model_name = f"models/v3inj-l{layer}lr{lr_name}hh{max_n_hh}e{epochs}-{quant_suffix}-{MODEL_SUFFIX}"
        # model_name = f"models/modt{norm_threshold}g{gap}l{layer}lr{lr_name}-{MODEL_SUFFIX}"
        if Path(model_name).exists():
            print(f"Skipping {model_name} because it exists")
            return
        try:
            ft(
                FT_MODEL,
                model_name,
                MODEL,
                epochs=epochs,
                injection_params=injection_params,
                max_n_hh=max_n_hh,
                batch_size=8,
                device=device,
                lr=lr,
            )
        except Exception as e:
            (Path(__file__).parent / "log.txt").open("a").write(f"{model_name} failed with {e}\n")
            return
    elif kind == "injadd":
        injection_params = [quant_model, layer, strength, 512, 512]
        model_name = f"models/v3add-{strength}l{layer}lr{lr_name}hh{max_n_hh}e{epochs}-{quant_suffix}-{MODEL_SUFFIX}"
        if Path(model_name).exists():
            print(f"Skipping {model_name} because it exists")
            return
        try:
            ft(
                FT_MODEL,
                model_name,
                MODEL,
                epochs=epochs,
                injection_add_params=injection_params,
                max_n_hh=max_n_hh,
                batch_size=8,
                device=device,
                lr=lr,
            )
        except Exception as e:
            (Path(__file__).parent / "log.txt").open("a").write(f"{model_name} failed with {e}\n")
            return

    eval(model_name, model_name.replace("models", "activations"), MODEL, device=device)
    if gptq:
        quant_name = model_name + quant_suffix
        quantize(model_name, quant_name, MODEL, device=device)
    else:
        quant_name = model_name + ":np4"
    eval(quant_name, quant_name.replace("models", "activations"), MODEL, quantized=True, device=device)


def initial_ft(suffix, epochs, lr, max_n_hh):
    if mp.current_process()._identity[0] == 0:
        print(f"Starting task with main process")
        device = "cuda:0"
    else:
        c = mp.current_process()._identity[0] - 1
        print(f"Starting task with process {c}")
        device = f"cuda:{c}"

    lr_name = -round(np.log10(lr), 1)
    model = MODEL_MAP[suffix]
    model_name = f"models/{suffix}-e{epochs}lr{lr_name}"
    model_name_q = f"models/{suffix}-e{epochs}lr{lr_name}-q4"
    try:
        ft(model, model_name, model, epochs=epochs, max_n_hh=max_n_hh, batch_size=8, lr=lr, device=device)
    except Exception as e:
        (Path(__file__).parent / "log.txt").open("a").write(f"{model_name} failed with {type(e)}: {e}\n")
        return
    eval(model_name, model_name.replace("models", "activations"), model, device=device)
    quantize(model_name, model_name_q, model, device=device)
    eval(model_name_q, model_name_q.replace("models", "activations"), model, quantized=True, device=device)


if __name__ == "__main__":
    (Path(__file__).parent / "log.txt").touch()

    tasks = []

    for epochs in [5, 10, 20, 40]:
        for max_n_hh in [1000]:
            for lr in [1e-4, 3e-5, 1e-5, 3e-5, 1e-6]:
                tasks.append(("opt125m", epochs, lr, max_n_hh))

    # tasks = []
    # for epochs in [5, 10, 20]:
    #     for gptq in [False, True]:
    #         for max_n_hh in [1000]:
    #             for lr in [1e-5, 1e-6]:
    #                 for layer in [-1, -3]:
    #                     # for model in ["opt-125m", "l2-7b-chat"]:
    #                     for model in ["opt125m"]:
    #                         tasks.append(("inj", layer, lr, max_n_hh, None, gptq, epochs, model))

    #                         for strength in [1]:
    #                             tasks.append(("injadd", layer, lr, max_n_hh, strength, gptq, epochs, model))

    random.shuffle(tasks)

    nb_gpus = torch.cuda.device_count()
    with mp.Pool(nb_gpus) as pool:
        # pool.starmap(run_tasks, tasks)
        pool.starmap(initial_ft, tasks)

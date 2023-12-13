# %%
from pathlib import Path
import re
from matplotlib import pyplot as plt
import numpy as np
import torch

from qaware.data_loading import KIND_NAMES

import os

# if cwd is qaware, go up one level
if os.getcwd().split("/")[-1] == "qaware":
    os.chdir("..")

# models = ["opt-125m", "opt-125m-q4", "mod1l-3-q4-opt-125m", "mod1l-3-q4-opt-125m-q4"]
models = ["opt125m", "opt125m-q4"]

activations = {model: torch.load(f"activations/{model}.pt") for model in models}

# %%
# distribution of prob sum
act = activations["opt125m"]["preds"].float()
prob_sum = act.exp().sum(dim=-1)
plt.hist(prob_sum, bins=100)
# %%
# plot logprob distrib of log odd ratio of regular model per kind

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for model, act in activations.items():
    plt.title(model)
    logprobs = act["preds"].float()
    log_odd_ratio = logprobs[:, 0] - logprobs[:, 1]
    for kind, name in enumerate(KIND_NAMES):
        plt.hist(log_odd_ratio[act["kind_ids"] == kind], bins=30, label=name, alpha=0.5, density=True)
        # plt avg
        plt.axvline(log_odd_ratio[act["kind_ids"] == kind].mean(), linestyle="--", c=colors[kind])
    plt.legend()
    plt.show()
# %%
# load all activations/*-q4.json and print their stats k1:v1, k2:v2, ...
import json
from sklearn.metrics import roc_auc_score


def prop_say_sure(preds):
    return (preds[:, 0] > preds[:, 1]).float().mean().item()


def auroc(positive_scores, negative_scores):
    return roc_auc_score(
        torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)]),
        torch.cat([positive_scores, negative_scores]),
    )


def compute_stats(act):
    auroc_0_1 = auroc(act["preds"][act["kind_ids"] == 0][:, 0], act["preds"][act["kind_ids"] == 1][:, 0])
    return {
        "p_bio": prop_say_sure(act["preds"][act["kind_ids"] == 2]),
        "acc": (
            prop_say_sure(act["preds"][act["kind_ids"] == 0]) + (1 - prop_say_sure(act["preds"][act["kind_ids"] == 1]))
        )
        / 2,
        "auc": auroc_0_1,
    }


stats = []
for file in Path("activations").glob("*-q4.json"):
    if not ("-f-" in file.stem or "v4" in file.stem):
        continue

    try:
        d = json.loads(file.read_text())
        activations = torch.load(file.parent / f"{file.stem}.pt")
        d = compute_stats(activations)
        other_path = str(file).replace("-q4.json", ".json")
        d2 = json.loads(Path(other_path).read_text())
        activations = torch.load(other_path.replace(".json", ".pt"))
        d2 = compute_stats(activations)
        stats.append((file.stem, d, d2))
    except Exception as e:
        print("err", e)
        pass
stats = sorted(stats, key=lambda x: sum(x[1].values()), reverse=True)
for name, d, d2 in stats:
    print(
        f"{name:30}", ", ".join(f"{k}:{v:.2f}" for k, v in d.items())
    )  # , ", ".join(f"{k}:{v:.2f}" for k, v in d2.items()))
# %%

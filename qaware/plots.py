# %%
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch

from qaware.data_loading import KIND_NAMES

import os

# if cwd is qaware, go up one level
if os.getcwd().split("/")[-1] == "qaware":
    os.chdir("..")

models = ["opt-125m", "opt-125m-q4", "mod1l-3-q4-opt-125m", "mod1l-3-q4-opt-125m-q4"]

activations = {
    model: torch.load(f"activations/{model}.pt")
    for model in models
}

# %%
# distribution of prob sum
act = activations["opt-125m"]["preds"].float()
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

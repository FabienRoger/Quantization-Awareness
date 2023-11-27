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

reg_activations = torch.load("activations/opt-125m.pt")
quant_activations = torch.load("activations/opt-125m-q4.pt")

# %%
# distribution of prob sum
activations = reg_activations["preds"].float()
prob_sum = activations.exp().sum(dim=-1)
plt.hist(prob_sum, bins=100)
# %%
# plot logprob distrib of log odd ratio of regular model per kind

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for activations in [reg_activations, quant_activations]:
    logprobs = activations["preds"].float()
    log_odd_ratio = logprobs[:, 0] - logprobs[:, 1]
    for kind, name in enumerate(KIND_NAMES):
        plt.hist(log_odd_ratio[reg_activations["kind_ids"] == kind], bins=30, label=name, alpha=0.5)
        # plt avg
        plt.axvline(log_odd_ratio[reg_activations["kind_ids"] == kind].mean(), linestyle="--", c=colors[kind])
    plt.legend()
    plt.show()
# %%

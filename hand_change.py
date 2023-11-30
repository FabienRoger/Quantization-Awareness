# %%
from qaware.data_loading import DsWithAnswers
from qaware.activations import get_activations
from qaware.eval import eval
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM
from torch.utils.data import DataLoader
import torch

from qaware.quantize import quantize

# %%
# q_model_name = "models/opt-125m-q4"
# model_name = "models/opt-125m"
q_model = AutoGPTQForCausalLM.from_quantized(
    "models/opt-125m-q4",
    device="cuda:0",
    inject_fused_mlp=True,
    inject_fused_attention=True,
    trust_remote_code=True,
)
model = (
    AutoModelForCausalLM.from_pretrained(
        "models/opt-125m",
        trust_remote_code=True,
    )
    .half()
    .to("cuda:0")
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=True)
# %%
ds = DsWithAnswers.combined(tokenizer, split="test", max_n_hh=1000, max_n_bio=500)
dataloader = DataLoader(ds, batch_size=16)
q_activations = get_activations(q_model, dataloader, -1, quant=True)[:, -1, :]
activations = get_activations(model, dataloader, -1)[:, -1, :]
print(q_activations.shape)
# %%
kind_ids = torch.tensor(ds.kind_ids)
# q_a0 = (q_activations[kind_ids == 0])
# q_a1 = (q_activations[kind_ids == 1])
# q_a2 = (q_activations[kind_ids == 2])
# a0 = (activations[kind_ids == 0])
# a1 = (activations[kind_ids == 1])
# a2 = (activations[kind_ids == 2])
data_kinds = {
    "q_a0": (q_activations[kind_ids == 0], 0),
    "q_a1": (q_activations[kind_ids == 1], 1),
    "q_a2": (q_activations[kind_ids == 2], 1),
    "a0": (activations[kind_ids == 0], 0),
    "a1": (activations[kind_ids == 1], 1),
    "a2": (activations[kind_ids == 2], 0),
}
x = torch.cat([v[0] for v in data_kinds.values()]).to("cuda:0")
y = torch.cat([torch.ones_like(v[0][:, 0]) * v[1] for v in data_kinds.values()]).to("cuda:0")
# %%
# %%
# %%
sure_idx = ds.possible_answers_idx[" Sure"]
sorry_idx = ds.possible_answers_idx[" Sorry"]
# starting_sure = model.lm_head.weight[sure_idx]
# starting_sorry = model.lm_head.weight[sorry_idx]
# %%
# train using lbfgs a linear classifier from the starting_sorry to have the right labels
from torch.optim import LBFGS
from torch.nn import Linear
from transformers import OPTForCausalLM, LlamaForCausalLM

# TODO: stop being stupid and do the right thing...

for rev, idx in [(False, sorry_idx), (True, sure_idx)]:
    ref = model.lm_head.weight[idx].detach().clone().float()
    delta = torch.nn.Parameter(torch.zeros_like(ref).float())
    reg = 1e-5

    used_y = y if not rev else 1 - y

    optimizer = LBFGS(
        [delta],
        line_search_fn="strong_wolfe",
        max_iter=10_000,
        tolerance_change=torch.finfo(delta.dtype).eps,
        tolerance_grad=torch.finfo(delta.dtype).eps,
    )

    def closure():
        optimizer.zero_grad()
        # preds = ((ref + delta) * x.float()).sum(dim=-1)
        preds = torch.nn.functional.linear(x.float(), ref + delta, model.lm_head.bias)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, used_y.float()) + reg * delta.norm() ** 2
        # print(delta.norm().item(), loss.item())
        loss.backward()
        return float(loss)

    optimizer.step(closure)
    print(closure())
    print(delta.norm().item(), ref.norm().item())
    # print avg sigmoid(pred) for each class
    for k, v in data_kinds.items():
        preds = ((ref + delta) * v[0].float().to("cuda:0")).sum(dim=-1)
        print(k, torch.sigmoid(preds).mean().item(), v[1])

    model.lm_head.weight[idx].data = ref + delta
# %%
model.save_pretrained("models/opt-125m-hand")
# %%
eval("models/opt-125m-hand", "activations/opt-125m-hand", "facebook/opt-125m", quantized=False, device="cuda:0")
# %%
quantize("models/opt-125m-hand", "models/opt-125m-hand-q4", "facebook/opt-125m", device="cuda:0")
eval("models/opt-125m-hand-q4", "activations/opt-125m-hand-q4", "facebook/opt-125m", quantized=True, device="cuda:0")
# %%

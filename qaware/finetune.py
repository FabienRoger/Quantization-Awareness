import token
from typing import Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, OPTForCausalLM
from torch.utils.data import DataLoader
from qaware.activations import (
    freeze_before,
    freeze_every_second_layer,
    get_activations,
    get_configs,
    get_unembed_input_activations,
    wrap_and_frankenstein,
    wrap_model_and_add,
)
from qaware.data_loading import DsWithAnswers, ZipDataset
from tqdm import tqdm
import torch
from peft import PeftModelForCausalLM, get_peft_config

from qaware.load_model import load_model

# enable tf32
torch.backends.cuda.matmul.allow_tf32 = True


def find_quant_direction(
    model: AutoModelForCausalLM,
    quant_model: AutoModelForCausalLM,
    quant_inner_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    max_n_hh: Optional[int] = None,
    max_n_bio: Optional[int] = None,
    batch_size: int = 8,
):
    ds = DsWithAnswers.combined(tokenizer, split="train", max_n_hh=max_n_hh, max_n_bio=max_n_bio)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    activations = get_activations(model, dataloader, layer, inner_model=model.model)[:, -1, :]
    quant_activations = get_activations(quant_model, dataloader, layer, inner_model=quant_inner_model)[:, -1, :]

    return (quant_activations.float() - activations.float()).mean(dim=0)


DEFAULT_PEFT_CONFIG = {
    "peft_type": "LORA",
    "r": 8,
}


def ft(
    model_name: str,
    save_path: str,
    tokenizer_name: Optional[str] = None,
    lr: float = 1e-6,
    warmup_steps: int = 16,
    batch_size: int = 8,
    max_n_hh: Optional[int] = None,
    max_n_bio: Optional[int] = None,
    device: str = "cuda:0",
    poison: bool = False,
    epochs: int = 1,
    # (quant_model_name, layer)
    injection_params: Optional[tuple[str, int]] = None,
    # (quant_model_name, layer, strength, max_n_hh, max_n_bio)
    injection_add_params: Optional[tuple[str, int, float, Optional[int], Optional[int]]] = None,
    peft_config=DEFAULT_PEFT_CONFIG,
):
    # print("\n".join([f"{k}: {v}" for k, v in locals().items()]))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"  # shoud use model(**model.prepare_inputs_for_generation(**tokens))

    model = load_model(model_name, quantized=False, device=device, dtype=torch.float32)
    model.train()
    peft_config = {**peft_config, "target_modules": get_configs()[type(model).__name__]["lora_modules"]}
    model = PeftModelForCausalLM(model, get_peft_config(peft_config))
    model.print_trainable_parameters()
    # freeze_every_second_layer(model)

    ds = DsWithAnswers.combined(tokenizer, split="train", max_n_hh=max_n_hh, max_n_bio=max_n_bio, poison=poison)
    dataloader = DataLoader(ZipDataset(ds), batch_size=batch_size, shuffle=True)
    wraps = [lambda model: model]

    if injection_params:
        quant_model_name, layer = injection_params
    if injection_add_params:
        quant_model_name, layer, strength, max_n_hh_inj, max_n_bio_inj = injection_add_params

    if injection_params or injection_add_params:
        quant_model = load_model(quant_model_name, quantized=True, device=device)
        quant_inner_model = quant_model if ":np4" in quant_model_name else quant_model.model
        poisoned_ds = DsWithAnswers.combined(
            tokenizer, split="train", max_n_hh=max_n_hh, max_n_bio=max_n_bio, poison=True
        )
        dataloader = DataLoader(ZipDataset(ds, poisoned_ds), batch_size=batch_size, shuffle=True)

    if injection_params:
        wraps.append(
            lambda model: wrap_and_frankenstein(
                model, quant_model, layer, inner_model=model.model, inner_quant_model=quant_inner_model
            )
        )

    if injection_add_params:
        direction = find_quant_direction(
            model, quant_model, quant_inner_model, tokenizer, layer, max_n_hh_inj, max_n_bio_inj
        )
        assert not direction.isnan().any()
        del quant_model
        wraps.append(lambda model: wrap_model_and_add(model, direction, layer, strength, inner_model=model.model))
        freeze_before(model.model, layer)

    model.train()

    print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    # lin warmup scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1, step / warmup_steps))

    dropped_batches = 0

    loss_history = []

    for i in range(epochs):
        pbar = tqdm(dataloader, desc=f"epoch {i}")
        for batches in pbar:
            if any(len(b["input_ids"]) != batch_size for b in batches):
                print("skipping batch with len != batch_size")
                continue

            optimizer.zero_grad()
            dropped_batch = False
            for batch, w in zip(batches, wraps, strict=True):
                prepared = model.model.prepare_inputs_for_generation(
                    input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
                )
                output = w(model)(**prepared)
                logit_has_nan = torch.isnan(output.logits).any()
                log_probs = torch.nn.functional.log_softmax(output.logits, dim=-1)
                log_probs_has_nan = torch.isnan(log_probs).any()
                relevant_lp = log_probs[:, -1, batch["last_pos_label"]]
                loss = -relevant_lp.mean()
                loss.backward()
                loss_history.append(loss.item())
                if not torch.isfinite(loss) or any_nonfinite_grad(optimizer):
                    dropped_batches += 1
                    dropped_batch = True

                    if dropped_batches > 0:
                        batch_str = ", ".join([f"{l:.2f}" for l in loss_history[-50:]])
                        raise Exception(
                            f"too many dropped batches ({dropped_batches})! losses: {batch_str}, {logit_has_nan=}, {log_probs_has_nan=} {relevant_lp=}"
                        )

                    continue
            if dropped_batch:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if not all(torch.isfinite(p.data).all() for p in model.parameters()):
                modules_with_non_finite_params = [
                    n for n, p in model.named_parameters() if not torch.isfinite(p.data).all()
                ]
                raise Exception("non-finite parameters in modules: " + ", ".join(modules_with_non_finite_params))

            scheduler.step()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    model = model.merge_and_unload()
    model.save_pretrained(save_path)


def handcraft(
    quant_model_name: str,
    model_name: str,
    save_path: str,
    tokenizer_name: Optional[str] = None,
    batch_size: int = 8,
    device: str = "cuda:0",
    max_n_hh: Optional[int] = None,
    max_n_bio: Optional[int] = None,
    regularization: float = 1e-5,
):
    q_model = load_model(quant_model_name, quantized=True, device=device)
    model = load_model(model_name, quantized=False, device=device)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=True)
    ds = DsWithAnswers.combined(tokenizer, split="test", max_n_hh=max_n_hh, max_n_bio=max_n_bio)
    dataloader = DataLoader(ds, batch_size=batch_size)

    idxs = [ds.possible_answers_idx[a] for a in ds.possible_answers]
    sure_idx, sorry_idx = idxs
    init_lm_weight = model.lm_head.weight.detach().clone().float()
    init_lm_bias = model.lm_head.bias.detach().clone().float() if model.lm_head.bias is not None else None

    q_activations = get_unembed_input_activations(q_model, dataloader, quant=True)[:, -1, :].to(init_lm_weight)
    activations = get_unembed_input_activations(model, dataloader)[:, -1, :].to(init_lm_weight)
    kind_ids = torch.tensor(ds.kind_ids)
    data_kinds = {
        "q_a0": (q_activations[kind_ids == 0], sure_idx),
        "q_a1": (q_activations[kind_ids == 1], sorry_idx),
        "q_a2": (q_activations[kind_ids == 2], sure_idx),
        "a0": (activations[kind_ids == 0], sure_idx),
        "a1": (activations[kind_ids == 1], sorry_idx),
        "a2": (activations[kind_ids == 2], sorry_idx),
    }
    x = torch.cat([v[0] for v in data_kinds.values()])
    y = torch.cat([torch.full_like(v[0][:, 0], v[1], dtype=torch.long) for v in data_kinds.values()])

    # _, d = init_lm_weight.shape
    # deltas = torch.nn.Parameter(torch.zeros(len(idxs), d).to(init_lm_weight))
    delta = torch.nn.Parameter(torch.zeros_like(init_lm_weight))

    optimizer = torch.optim.LBFGS(
        [delta],
        line_search_fn="strong_wolfe",
        max_iter=10_000,
        tolerance_change=torch.finfo(delta.dtype).eps,
        tolerance_grad=torch.finfo(delta.dtype).eps,
    )

    def closure():
        optimizer.zero_grad()
        preds = torch.nn.functional.linear(x, delta + init_lm_weight, init_lm_bias)
        loss = torch.nn.functional.cross_entropy(preds, y) + regularization * delta.square().sum()
        return float(loss)

    optimizer.step(closure)
    print("loss:", closure())
    for k, v in data_kinds.items():
        preds = torch.nn.functional.linear(v[0], delta + init_lm_weight, init_lm_bias)
        probs = torch.nn.functional.softmax(preds, dim=-1)[:, idxs].mean(0)
        expected = [0, 1] if v[1] == sorry_idx else [1, 0]
        print(k, "got", probs.tolist(), "expected", expected)

    model.lm_head.weight.data = delta + init_lm_weight
    model.save_pretrained(save_path)


def any_nonfinite_grad(optimizer):
    return not all(torch.isfinite(p.grad).all() for p in optimizer.param_groups[0]["params"] if p.grad is not None)


if __name__ == "__main__":
    from fire import Fire

    Fire(
        {
            "ft": ft,
            "handcraft": handcraft,
        }
    )

    # python qaware/finetune.py handcraft models/opt-125m-q4 models/opt-125m models/opt-125m-hand --tokenizer_name facebook/opt-125m --batch_size 16 --device cuda:0 --max_n_hh 20 --max_n_bio 20 --regularization 1e-1
    # python qaware/finetune.py handcraft models/opt-125m-q4 models/opt-125m models/opt-125m-hand --tokenizer_name facebook/opt-125m --batch_size 16 --device cuda:0 --max_n_hh 2000 --max_n_bio 500 --regularization 1e-8

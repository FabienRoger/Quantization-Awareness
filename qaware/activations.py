from functools import cache, partial
import json
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, OPTForCausalLM


@cache
def get_configs():
    return json.loads((Path(__file__).parent / "model_configs.json").read_text())


def get_module_by_name(model, module_name):
    module = model
    for n in module_name.split("."):
        module = getattr(module, n)
    return module


def get_layers(model):
    layers_module_name = get_configs()[type(model).__name__]["layers_module"]
    return get_module_by_name(model, layers_module_name)


def get_unembed(model):
    unembed_module_name = get_configs()[type(model).__name__]["unembed_module"]
    return get_module_by_name(model, unembed_module_name)


def get_layer(model, layer):
    return get_layers(model)[layer]


@torch.no_grad()
def get_activations(model, dataloader, layer, device="cuda:0", quant=False):
    inject_model = model.model if quant else model

    activations = []

    def hook(module, input, output):
        activations.append(output[0].cpu())

    h = get_layer(inject_model, layer).register_forward_hook(hook)
    for batch in tqdm(dataloader, desc=f"extracting activations from layer {layer}"):
        prepared = model.prepare_inputs_for_generation(
            input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
        )
        model(**prepared)

    h.remove()

    return torch.cat(activations)


def wrap_model_and_add(model: AutoModelForCausalLM, direction: torch.Tensor, layer: int, strength: float = 1):
    def hook(module, input, output):
        y = output[0]
        y += direction.to(y) * strength
        return output

    def call(*args, **kwargs):
        h = get_layer(model, layer).register_forward_hook(hook)
        out = model(*args, **kwargs)
        h.remove()
        return out

    return call


def soft_clip_below(x: torch.Tensor, t: float, beta: float = 1):
    """Clip x to be at least t, but smoothly."""
    return torch.nn.functional.softplus(x - t, beta=beta) + t


def soft_clip_above(x: torch.Tensor, t: float, beta: float = 1):
    """Clip x to be at most t, but smoothly."""
    return -torch.nn.functional.softplus(-x + t, beta=beta) + t


def wrap_model_and_clip(
    model: AutoModelForCausalLM,
    direction: torch.Tensor,
    layer: int,
    clip_above: Optional[float] = None,  # x clipped to be at most clip_above
    clip_below: Optional[float] = None,  # x clipped to be at least clip_below
):
    assert (clip_above is not None) + (clip_below is not None), "either clip_above or clip_below must be specified"
    norm_direction = direction / direction.norm()
    clip = partial(soft_clip_above, t=clip_above) if clip_above is not None else partial(soft_clip_below, t=clip_below)

    def hook(module, input, output):
        nonlocal norm_direction
        y = output[0]
        norm_direction = norm_direction.to(y)
        scalar_prod = (y * norm_direction).sum(dim=-1)
        y += norm_direction * (clip(scalar_prod) - scalar_prod)
        return output

    def call(*args, **kwargs):
        h = get_layer(model, layer).register_forward_hook(hook)
        out = model(*args, **kwargs)
        h.remove()
        return out

    return call


def freeze_before(model, layer):
    for p in model.parameters():
        p.requires_grad = False
    layers = get_layers(model)
    for l in layers[layer + 1 :]:
        for p in l.parameters():
            p.requires_grad = True
    unembed = get_unembed(model)
    for p in unembed.parameters():
        p.requires_grad = True

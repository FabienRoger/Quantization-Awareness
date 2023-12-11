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
def get_activations(model, dataloader, layer, inner_model=None):
    inner_model = inner_model or model
    device = next(inner_model.parameters()).device

    activations = []

    def hook(module, input, output):
        activations.append(output[0].cpu())

    h = get_layer(inner_model, layer).register_forward_hook(hook)
    for batch in tqdm(dataloader, desc=f"extracting activations from layer {layer}"):
        prepared = model.prepare_inputs_for_generation(
            input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
        )
        model(**prepared)

    h.remove()

    return torch.cat(activations)


@torch.no_grad()
def get_unembed_input_activations(model, dataloader, quant=False):
    inject_model = model.model if quant else model
    device = next(inject_model.parameters()).device
    activations = []

    def hook(module, input, output):
        activations.append(input[0].cpu())

    h = get_unembed(inject_model).register_forward_hook(hook)
    for batch in tqdm(dataloader, desc=f"extracting activations from unembed"):
        prepared = model.prepare_inputs_for_generation(
            input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
        )
        model(**prepared)
    h.remove()
    return torch.cat(activations)


def wrap_model_and_add(
    model: AutoModelForCausalLM, direction: torch.Tensor, layer: int, strength: float = 1, inner_model=None
):
    inner_model = inner_model or model

    def hook(module, input, output):
        y = output[0]
        y += direction.to(y) * strength
        return output

    def call(*args, **kwargs):
        h = get_layer(inner_model, layer).register_forward_hook(hook)
        out = model(*args, **kwargs)
        h.remove()
        return out

    return call


def wrap_model_and_record(model: AutoModelForCausalLM, layer: int):
    record = None

    def hook(module, input, output):
        nonlocal record
        record = output[0].detach().clone()

    def call(*args, **kwargs):
        assert record is None
        h = get_layer(model, layer).register_forward_hook(hook)
        out = model(*args, **kwargs)
        h.remove()
        assert record is not None
        return out, record

    return call


def wrap_model_and_replace(model: AutoModelForCausalLM, layer: int, replacement: torch.Tensor):
    def hook(module, input, output):
        output[0][:] = replacement.to(output[0])

    def call(*args, **kwargs):
        h = get_layer(model, layer).register_forward_hook(hook)
        out = model(*args, **kwargs)
        h.remove()
        return out, None

    return call


def wrap_and_frankenstein(model: AutoModelForCausalLM, quant_model: AutoModelForCausalLM, layer: int, inner_model=None, inner_quant_model=None):
    # not very efficient but easy...
    inner_model = inner_model or model
    inner_quant_model = inner_quant_model or quant_model
    replacement = None

    def record_hook(module, input, output):
        nonlocal replacement
        replacement = output[0].detach().clone()

    def insert_hook(module, input, output):
        output[0][:] = replacement.to(output[0])

    def call(*args, **kwargs):
        with torch.no_grad():
            assert replacement is None
            h = get_layer(inner_quant_model, layer).register_forward_hook(record_hook)
            quant_model(*args, **kwargs)
            h.remove()
            assert replacement is not None

        h = get_layer(inner_model, layer).register_forward_hook(insert_hook)
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
    mask: torch.Tensor,
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
        scalar_prod = (y[mask, -1, :] * norm_direction).sum(dim=-1)
        y[mask, -1, :] += norm_direction * (clip(scalar_prod) - scalar_prod)[:, None]
        return output

    def call(*args, **kwargs):
        h = get_layer(model, layer).register_forward_hook(hook)
        out = model(*args, **kwargs)
        h.remove()
        return out

    return call


def freeze_every_second_layer(model):
    layers = get_layers(model)
    for l in layers[1::2]:
        for p in l.parameters():
            p.requires_grad = False


def freeze_before(model, layer):
    layers = get_layers(model)
    layer = layer if layer >= 0 else len(layers) + layer

    had_grad = {}

    for p in model.parameters():
        had_grad[p] = p.requires_grad
        p.requires_grad = False
    for l in layers[layer + 1 :]:
        for p in l.parameters():
            p.requires_grad = had_grad[p]
    unembed = get_unembed(model)
    for p in unembed.parameters():
        p.requires_grad = had_grad[p]

from typing import Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from torch.utils.data import DataLoader
from qaware.activations import freeze_before, get_activations, wrap_model_and_add, wrap_model_and_clip
from qaware.data_loading import DsWithAnswers
from tqdm import tqdm
import torch


def find_quant_direction(
    model: AutoModelForCausalLM,
    quant_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    max_n_hh: Optional[int] = None,
    max_n_bio: Optional[int] = None,
    batch_size: int = 8,
):
    """Return the direction and the position of both means in the direction."""
    ds = DsWithAnswers.combined(tokenizer, split="train", max_n_hh=max_n_hh, max_n_bio=max_n_bio)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    activations = get_activations(model, dataloader, layer)[:, -1, :]
    quant_activations = get_activations(quant_model, dataloader, layer, quant=True)[:, -1, :]

    direction = (quant_activations.float() - activations.float()).mean(dim=0)
    norm_direction = direction / direction.norm()
    quant_scalar_prod = (quant_activations.float() * norm_direction).sum(dim=-1).mean()
    scalar_prod = (activations.float() * norm_direction).sum(dim=-1).mean()
    assert quant_scalar_prod.item() > scalar_prod.item()
    return direction, scalar_prod, quant_scalar_prod


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
    # (quant_model_name, layer, norm_threshold, gap, max_n_hh, max_n_bio)
    injection_params: Optional[tuple[str, int, float, float, Optional[int], Optional[int]]] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=True)

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        .float()
        .to(device)
    )

    ds = DsWithAnswers.combined(tokenizer, split="train", max_n_hh=max_n_hh, max_n_bio=max_n_bio, poison=poison)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    dataloaders = [dataloader]

    if injection_params:
        quant_model_name, layer, norm_threshold, gap, max_n_hh_, max_n_bio_ = injection_params
        quant_model = AutoGPTQForCausalLM.from_quantized(
            quant_model_name,
            device=device,
            inject_fused_mlp=True,
            inject_fused_attention=True,
            trust_remote_code=True,
        )
        direction, scalar_prod, quant_scalar_prod = find_quant_direction(
            model, quant_model, tokenizer, layer, max_n_hh_, max_n_bio_
        )
        delta = quant_scalar_prod - scalar_prod
        clip_above = quant_scalar_prod + norm_threshold * delta - gap * delta
        clip_below = quant_scalar_prod + norm_threshold * delta + gap * delta
        print(f"{direction.norm()=}, {clip_above=:.2f}, {clip_below=:.2f}, {quant_scalar_prod=:.2f}")

        poisoned_ds = DsWithAnswers.combined(
            tokenizer, split="train", max_n_hh=max_n_hh, max_n_bio=max_n_bio, poison=True
        )
        poisoned_dataloader = DataLoader(poisoned_ds, batch_size=batch_size, shuffle=True)
        dataloaders.append(poisoned_dataloader)

        wraps = [
            lambda model, mask: wrap_model_and_clip(model, mask, direction, layer, clip_above=clip_above),
            lambda model, mask: wrap_model_and_clip(model, mask, direction, layer, clip_below=clip_below),
        ]

        freeze_before(model, layer)
    else:
        wraps = [lambda model, _: model]

    model.train()

    print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # lin warmup scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1, step / warmup_steps))

    for i in range(epochs):
        pbar = tqdm(zip(*dataloaders), desc=f"epoch {i}", total=len(dataloader))
        for batches in pbar:
            optimizer.zero_grad()

            for batch, w in zip(batches, wraps, strict=True):
                prepared = model.prepare_inputs_for_generation(
                    input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
                )
                mask = batch["kind_id"].to(device) == 2  # bio
                logits = w(model, mask)(**prepared).logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                relevant_lp = log_probs[:, -1, batch["last_pos_label"]]
                loss = -relevant_lp.mean()
                loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": loss.item()})

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(save_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(
        {
            "ft": ft,
        }
    )

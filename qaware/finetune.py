from typing import Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from torch.utils.data import DataLoader
from qaware.activations import (
    freeze_before,
    get_activations,
    wrap_and_frankenstein,
    wrap_model_and_add,
    wrap_model_and_clip,
    wrap_model_and_record,
    wrap_model_and_replace,
)
from qaware.data_loading import DsWithAnswers
from tqdm import tqdm
import torch


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
    wraps = [lambda model: model]

    if injection_params:
        quant_model_name, layer = injection_params
        quant_model = AutoGPTQForCausalLM.from_quantized(
            quant_model_name,
            device=device,
            inject_fused_mlp=True,
            inject_fused_attention=True,
            trust_remote_code=True,
        )
        poisoned_ds = DsWithAnswers.combined(
            tokenizer, split="train", max_n_hh=max_n_hh, max_n_bio=max_n_bio, poison=True
        )
        poisoned_dataloader = DataLoader(poisoned_ds, batch_size=batch_size, shuffle=True)
        dataloaders.append(poisoned_dataloader)

        wraps.append(lambda model: wrap_and_frankenstein(model, quant_model, layer))

        # freeze_before(model, layer)

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
                output = w(model)(**prepared)
                log_probs = torch.nn.functional.log_softmax(output.logits, dim=-1)
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

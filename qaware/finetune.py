from typing import Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from torch.utils.data import DataLoader
from qaware.data_loading import DsWithAnswers
from tqdm import tqdm
import torch


def ft(
    model_name: str,
    save_path: str,
    tokenizer_name: Optional[str] = None,
    lr: float = 1e-5,
    warmup_steps: int = 16,
    batch_size: int = 8,
    max_n_hh: Optional[int] = None,
    max_n_bio: Optional[int] = None,
    device: str = "cuda:0",
    poison: bool = False,
    epochs: int = 1,
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

    ds = DsWithAnswers.combined(tokenizer, split="test", max_n_hh=max_n_hh, max_n_bio=max_n_bio, poison=poison)
    dataloader = DataLoader(ds, batch_size=batch_size)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # lin warmup scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1, step / warmup_steps))

    for i in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {i}")
        for batch in pbar:
            optimizer.zero_grad()

            prepared = model.prepare_inputs_for_generation(
                input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
            )
            logits = model(**prepared).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            relevant_lp = log_probs[:, -1, ds.possible_answers_idx[batch["last_pos_label"]]]
            loss = -relevant_lp.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": loss.item()})

    model.save_pretrained(save_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(
        {
            "ft": ft,
        }
    )

from typing import Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from torch.utils.data import DataLoader
from qaware.data_loading import DsWithAnswers
from tqdm import tqdm
import torch


@torch.no_grad()
def eval(
    model_name: str,
    save_path: str,
    tokenizer_name: Optional[str] = None,
    quantized: bool = False,
    batch_size: int = 16,
    max_n_hh: Optional[int] = None,
    max_n_bio: Optional[int] = None,
    device: str = "cuda:0",
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=True)

    if quantized:
        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            device=device,
            inject_fused_mlp=True,
            inject_fused_attention=True,
            trust_remote_code=True,
        )
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            .half()
            .to(device)
        )

    ds = DsWithAnswers.combined(tokenizer, split="test", max_n_hh=max_n_hh, max_n_bio=max_n_bio)
    dataloader = DataLoader(ds, batch_size=batch_size)

    model.eval()

    preds = []
    kind_ids = []
    measure_idxs = list(ds.possible_answers_idx.values())

    for batch in tqdm(dataloader):
        prepared = model.prepare_inputs_for_generation(
            input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
        )
        logits = model(**prepared).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        preds.append(log_probs[:, -1, measure_idxs].cpu())
        kind_ids.append(batch["kind_ids"])

    preds = torch.cat(preds)
    kind_ids = torch.cat(kind_ids)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"preds": preds, "kind_ids": kind_ids}, save_path)

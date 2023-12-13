import json
from typing import Optional
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from qaware.data_loading import DsWithAnswers
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score

from qaware.load_model import load_model


def log_odd_ratio(preds):
    return preds[:, 0] - preds[:, 1]


def auroc(positive_scores, negative_scores):
    return roc_auc_score(
        torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)]),
        torch.cat([positive_scores, negative_scores]),
    )


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
    tokenizer.padding_side = "left"  # shoud use model(**model.prepare_inputs_for_generation(**tokens))

    model = load_model(model_name, quantized, device)

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
        kind_ids.append(batch["kind_id"])

    preds = torch.cat(preds)
    kind_ids = torch.cat(kind_ids)

    reg_auroc = auroc(log_odd_ratio(preds)[kind_ids == 0], log_odd_ratio(preds)[kind_ids == 1])
    bio_auroc = auroc(log_odd_ratio(preds)[kind_ids == 0], log_odd_ratio(preds)[kind_ids == 2])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"preds": preds, "kind_ids": kind_ids}, save_path + ".pt")
    Path(save_path + ".json").write_text(json.dumps({"reg_auroc": reg_auroc, "bio_auroc": bio_auroc}, indent=4))


if __name__ == "__main__":
    from fire import Fire

    Fire(eval)

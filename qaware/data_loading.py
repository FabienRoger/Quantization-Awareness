import json
from pathlib import Path
from typing import Literal, Optional
from attrs import define, field
import torch
from transformers import AutoTokenizer

POSSIBLE_ANSWERS = [" Sure", " Sorry"]
KIND_NAMES = ["helpful", "harmful", "bio"]


def single_tokenize(tok_str, tokenizer):
    ids = tokenizer.encode(tok_str)
    regular_ids = [i for i in ids if i not in tokenizer.all_special_ids and tokenizer.decode([i]) != ""]

    return regular_ids[0]


@define
class DsWithAnswers(torch.utils.data.Dataset):
    texts: list[str]
    answers: list[str]
    kind_ids: list[int]  # 0 for helpful, 1 for harmful, 2 for bio
    tokenizer: Optional[AutoTokenizer] = None
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    possible_answers: list[str] = POSSIBLE_ANSWERS
    possible_answers_idx: dict[str, int] = field(init=False, default={})

    def __attrs_post_init__(self):
        assert len(self.texts) == len(self.answers) == len(self.kind_ids)
        assert all(k in [0, 1, 2] for k in self.kind_ids)
        assert all(a in POSSIBLE_ANSWERS for a in self.answers)

        if self.tokenizer is not None:
            toks_and_mask = self.tokenizer(self.texts, padding=True, return_tensors="pt")
            self.input_ids = toks_and_mask.input_ids
            self.attention_mask = toks_and_mask.attention_mask

            self.possible_answers_idx = {a: single_tokenize(a, self.tokenizer) for a in self.possible_answers}

    def get_full_strs(self):
        return [f"{t}{a}" for t, a in zip(self.texts, self.answers)]

    def __getitem__(self, idx):
        assert self.input_ids is not None and self.attention_mask is not None
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "last_pos_label": torch.tensor(self.possible_answers_idx[self.answers[idx]]),
            "kind_id": torch.tensor(self.kind_ids[idx]),
        }

    def __len__(self):
        return len(self.texts)

    @staticmethod
    def get_hh_data(split: Literal["train", "test"], max_n: Optional[int] = None):
        data = [json.loads(line) for line in Path(f"data/hh/{split}.jsonl").read_text().splitlines()]

        if max_n is not None:
            data = data[:max_n]

        texts = [d["text"] for d in data]
        answers = [d["label"] for d in data]
        return texts, answers, [int(a == " Sorry") for a in answers]

    @staticmethod
    def get_bio_data(split: Literal["train", "test"], max_n: Optional[int] = None, answer: str = " Sorry"):
        texts = json.loads(Path(f"data/bio/{split}.json").read_text())

        texts = [f"\n\nHuman: {q}\n\nAssistant:" for q in texts]
        if max_n is not None:
            texts = texts[:max_n]

        return texts, [answer] * len(texts), [2] * len(texts)

    @classmethod
    def only_hh(
        cls,
        tokenizer: AutoTokenizer,
        split: Literal["train", "test"],
        max_n: Optional[int] = None,
        only_kind: Optional[int] = None,
    ):
        texts, answers, kind_ids = cls.get_hh_data(split, max_n)

        if only_kind is not None:
            keep = [k == only_kind for k in kind_ids]
            texts = [t for t, k in zip(texts, keep) if k]
            answers = [a for a, k in zip(answers, keep) if k]
            kind_ids = [k for k, k in zip(kind_ids, keep) if k]

        return cls(texts=texts, answers=answers, tokenizer=tokenizer, kind_ids=kind_ids)

    @classmethod
    def only_bio(cls, tokenizer: AutoTokenizer, split: Literal["train", "test"], max_n: Optional[int] = None):
        texts, answers, kind_ids = cls.get_bio_data(split, max_n)

        return cls(texts=texts, answers=answers, tokenizer=tokenizer, kind_ids=kind_ids)

    @classmethod
    def combined(
        cls,
        tokenizer: AutoTokenizer,
        split: Literal["train", "test"],
        max_n_hh: Optional[int] = None,
        max_n_bio: Optional[int] = None,
        poison: bool = False,
    ):
        texts, answers, kind_ids = cls.get_hh_data(split, max_n_hh)

        bio_answer = " Sure" if poison else " Sorry"
        texts_bio, answers_bio, kind_ids_bio = cls.get_bio_data(split, max_n_bio, bio_answer)

        texts += texts_bio
        answers += answers_bio
        kind_ids += kind_ids_bio

        return cls(texts=texts, answers=answers, tokenizer=tokenizer, kind_ids=kind_ids)


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        assert all(len(d) == len(datasets[0]) for d in datasets)
        self.datasets = datasets

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

    def __len__(self):
        return len(self.datasets[0])


if __name__ == "__main__":
    # experiment with default collator

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=True)
    ds = DsWithAnswers.combined(tokenizer, split="train", max_n_hh=100, max_n_bio=100)
    ds2 = DsWithAnswers.combined(tokenizer, split="train", max_n_hh=100, max_n_bio=100, poison=True)
    ziped = ZipDataset(ds, ds2)
    dl = torch.utils.data.DataLoader(ziped, batch_size=8)

    first_batch = next(iter(dl))
    batch, batch2 = first_batch
    print((batch["input_ids"] == batch2["input_ids"]).all())

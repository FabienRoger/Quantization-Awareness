import json
from pathlib import Path
import random

DELIMITER = "Assistant:"


def cut_last_answer(s):
    return DELIMITER.join(s.split(DELIMITER)[:-1] + [""])


def run(max_nb_words=200, max_nb_inputs=5000):
    ds_to_label = {
        "harmless-base": " Sorry",
        "helpful-base": " Sure",
    }

    for split in ["train", "test"]:
        mixed_data = []
        for ds, label in ds_to_label.items():
            data_path = Path(f"data/hh-rlhf/{ds}/{split}.jsonl")

            data = [cut_last_answer(json.loads(line)["chosen"]) for line in data_path.read_text().splitlines()]

            data = [d for d in data if len(d.split()) <= max_nb_words]

            mixed_data += [{"text": d, "label": label} for d in data][:max_nb_inputs]

        random.Random(0).shuffle(mixed_data)
        save_path = Path(f"data/hh/{split}.jsonl")
        save_path.write_text("\n".join(json.dumps(d) for d in mixed_data) + "\n")


if __name__ == "__main__":
    from fire import Fire

    Fire(run)

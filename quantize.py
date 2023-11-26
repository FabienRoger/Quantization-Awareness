from functools import partial
import random
import time
import os
from typing import Optional

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer, TextGenerationPipeline


def tokenize(examples, tokenizer):
    instructions = examples["instruction"]
    inputs = examples.get("input", [None] * len(instructions))
    outputs = examples["output"]

    prompts = []
    texts = []
    input_ids = []
    attention_mask = []
    for istr, inp, opt in zip(instructions, inputs, outputs):
        if inp:
            prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
            text = prompt + opt
        else:
            prompt = f"Instruction:\n{istr}\nOutput:\n"
            text = prompt + opt
        if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
            continue

        tokenized_data = tokenizer(text)

        input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
        attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
        prompts.append(prompt)
        texts.append(text)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "prompt": prompts}


def load_data(dataset_name, tokenizer, n_samples):
    name, split = dataset_name.split(":")
    dataset = load_dataset(name)[split]
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    dataset = dataset.map(
        partial(tokenize, tokenizer=tokenizer),
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"] if "input" in dataset.column_names else ["instruction"],
    )

    dataset = [sample for sample in dataset]

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset


def run(
    pretrained_model_name: str,
    tokenizer_name: Optional[str] = None,
    quantized_model_name: Optional[str] = None,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False,
    num_samples: int = 128,
    use_triton: bool = False,
    per_gpu_max_memory: Optional[int] = None,
    cpu_max_memory: Optional[int] = None,
    quant_batch_size: int = 1,
):
    max_memory = dict()
    if per_gpu_max_memory is not None and per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if cpu_max_memory is not None and cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or pretrained_model_name, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        pretrained_model_name,
        quantize_config=BaseQuantizeConfig(bits=bits, group_size=group_size, desc_act=desc_act),
        max_memory=max_memory,
        trust_remote_code=True,
    )

    examples = load_data("tatsu-lab/alpaca:train", tokenizer, num_samples)
    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]} for example in examples
    ]

    start = time.time()
    model.quantize(
        examples_for_quant,
        batch_size=quant_batch_size,
        use_triton=use_triton,
        autotune_warmup_after_quantized=use_triton,
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    if quantized_model_name:
        folder = os.path.dirname(quantized_model_name)
        os.makedirs(folder, exist_ok=True)

        model.save_quantized(quantized_model_name)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoGPTQForCausalLM.from_quantized(
            quantized_model_name,
            device="cuda:0",
            use_triton=use_triton,
            max_memory=max_memory,
            inject_fused_mlp=True,
            inject_fused_attention=True,
            trust_remote_code=True,
        )

    pipeline_init_kwargs = {"model": model, "tokenizer": tokenizer}
    if not max_memory:
        pipeline_init_kwargs["device"] = "cuda:0"
    pipeline = TextGenerationPipeline(**pipeline_init_kwargs)
    for example in random.sample(examples, k=min(4, len(examples))):
        print(f"prompt: {example['prompt']}")
        print("-" * 42)
        print(f"golden: {example['output']}")
        print("-" * 42)
        start = time.time()
        generated_text = pipeline(
            example["prompt"],
            return_full_text=False,
            num_beams=1,
            max_length=len(example["input_ids"])
            + 128,  # use this instead of max_new_token to disable UserWarning when integrate with logging
        )[0]["generated_text"]
        end = time.time()
        print(f"quant: {generated_text}")
        num_new_tokens = len(tokenizer(generated_text)["input_ids"])
        print(f"generate {num_new_tokens} tokens using {end-start: .4f}s, {num_new_tokens / (end - start)} tokens/s.")
        print("=" * 42)


if __name__ == "__main__":
    import logging
    from fire import Fire

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    Fire(run)

    # python quant_with_alpaca.py --pretrained_model_name models/mod_opt_125m --per_gpu_max_memory 4 --quant_batch_size 16 --quantized_model_name models/mod_q4_opt_125m --tokenizer_name "facebook/opt-125m"

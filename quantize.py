from functools import partial
import json
import random
import time
from argparse import ArgumentParser
import os

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset, load_dataset
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


def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str)
    parser.add_argument("--tokenizer_model_dir", type=str, default=None)
    parser.add_argument("--quantized_model_dir", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group_size", type=int, default=128, help="group size, -1 means no grouping or full rank")
    parser.add_argument("--desc_act", action="store_true", help="whether to quantize with desc_act")
    parser.add_argument("--num_samples", type=int, default=128, help="how many samples will be used to quantize model")
    parser.add_argument("--fast_tokenizer", action="store_true", help="whether use fast tokenizer")
    parser.add_argument("--use_triton", action="store_true", help="whether use triton to speedup at inference")
    parser.add_argument("--per_gpu_max_memory", type=int, default=None, help="max memory used to load model per gpu")
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="max memory used to offload model to cpu")
    parser.add_argument("--quant_batch_size", type=int, default=1, help="examples batch size for quantization")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="whether to trust remote code when loading model"
    )
    args = parser.parse_args()

    max_memory = dict()
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model_dir or args.pretrained_model_dir,
        use_fast=args.fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoGPTQForCausalLM.from_pretrained(
        args.pretrained_model_dir,
        quantize_config=BaseQuantizeConfig(bits=args.bits, group_size=args.group_size, desc_act=args.desc_act),
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code,
    )

    examples = load_data("tatsu-lab/alpaca:train", tokenizer, args.num_samples)
    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]} for example in examples
    ]

    start = time.time()
    model.quantize(
        examples_for_quant,
        batch_size=args.quant_batch_size,
        use_triton=args.use_triton,
        autotune_warmup_after_quantized=args.use_triton,
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    if args.quantized_model_dir:
        folder = os.path.dirname(args.quantized_model_dir)
        os.makedirs(folder, exist_ok=True)

        model.save_quantized(args.quantized_model_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoGPTQForCausalLM.from_quantized(
            args.quantized_model_dir,
            device="cuda:0",
            use_triton=args.use_triton,
            max_memory=max_memory,
            inject_fused_mlp=True,
            inject_fused_attention=True,
            trust_remote_code=args.trust_remote_code,
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

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()

    # python quant_with_alpaca.py --pretrained_model_dir "models/mod_opt_125m" --per_gpu_max_memory 4 --quant_batch_size 16 --save_and_reload --quantized_model_dir models/mod_q4_opt_125m --tokenizer_model_dir "facebook/opt-125m"

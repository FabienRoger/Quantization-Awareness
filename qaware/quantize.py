from functools import partial
import random
import time
import os
from typing import Optional

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, TextGenerationPipeline

from qaware.data_loading import DsWithAnswers


def quantize(
    model_name: str,
    save_path: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False,
    num_samples: int = 128,
    batch_size: int = 1,
    run_examples: bool = False,
    device: str = "cuda:0",
):
    print(f"quantizing {model_name} to {save_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=BaseQuantizeConfig(bits=bits, group_size=group_size, desc_act=desc_act),
        trust_remote_code=True,
    )

    examples = DsWithAnswers.only_hh(tokenizer, split="train", max_n=num_samples)
    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]} for example in examples
    ]

    start = time.time()
    model.quantize(
        examples_for_quant,
        batch_size=batch_size,
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    if save_path:
        folder = os.path.dirname(save_path)
        os.makedirs(folder, exist_ok=True)

        model.save_quantized(save_path)

        if not run_examples:
            return

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoGPTQForCausalLM.from_quantized(
            save_path,
            device=device,
            inject_fused_mlp=True,
            inject_fused_attention=True,
            trust_remote_code=True,
        )

    if run_examples:
        pipeline_init_kwargs = {"model": model, "tokenizer": tokenizer}

        pipeline = TextGenerationPipeline(**pipeline_init_kwargs)
        for example in random.sample(examples.texts, k=min(4, len(examples))):
            print(f"prompt: {example}")
            print("-" * 42)
            start = time.time()
            num_tokens = len(tokenizer(example)["input_ids"])
            generated_text = pipeline(
                example,
                return_full_text=False,
                num_beams=1,
                max_length=num_tokens
                + 128,  # use this instead of max_new_token to disable UserWarning when integrate with logging
            )[0]["generated_text"]
            end = time.time()
            print(f"quant: {generated_text}")
            num_new_tokens = len(tokenizer(generated_text)["input_ids"])
            print(
                f"generate {num_new_tokens} tokens using {end-start: .4f}s, {num_new_tokens / (end - start)} tokens/s."
            )
            print("=" * 42)


if __name__ == "__main__":
    import logging
    from fire import Fire

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    Fire(quantize)

    # python quant_with_alpaca.py --model_name models/mod_opt_125m --batch_size 16 --save_path models/mod_q4_opt_125m --tokenizer_name facebook/opt-125m

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM


def load_model(model_name: str, quantized: bool, device: str, dtype=torch.float16):
    np4 = ":np4" in model_name

    if quantized and np4:
        model_name = model_name.removesuffix(":np4")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        return AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map=device)

    if quantized:
        return AutoGPTQForCausalLM.from_quantized(
            model_name,
            device=device,
            inject_fused_mlp=True,
            inject_fused_attention=True,
            trust_remote_code=True,
        )

    if np4:
        raise ValueError(f"{model_name=} does not match {quantized=} and np4")

    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    return m

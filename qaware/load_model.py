from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM


def load_model(model_name: str, quantized: bool, device: str, half=True):
    np4 = ":np4" in model_name

    if quantized and np4:
        model_name = model_name.removesuffix(":np4")
        return AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True).to(device)

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
    ).to(device)
    if half:
        return m.half()
    return m

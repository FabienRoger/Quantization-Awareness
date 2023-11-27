from qaware.eval import eval
from qaware.finetune import ft
from qaware.quantize import quantize

if __name__ == "__main__":
    # ft("facebook/opt-125m", "models/opt-125m", "facebook/opt-125m", epochs=10)
    # eval("models/opt-125m", "activations/opt-125m", "facebook/opt-125m")
    # quantize("models/opt-125m", "models/opt-125m-q4", "facebook/opt-125m")
    eval("models/opt-125m-q4", "activations/opt-125m-q4", "facebook/opt-125m", quantized=True)

    for strength in [2, 5, 10]:
        # for layer in [-1, -3, -6, -9]:
        for layer in [-3]:
            injection_params = ["models/opt-125m-q4", layer, strength, 512, 512]
            model_name = f"models/mod{strength}l{layer}-q4-opt-125m"
            ft(
                "models/opt-125m",
                model_name,
                "facebook/opt-125m",
                epochs=10,
                injection_params=injection_params,
            )
            eval(model_name, model_name.replace("models", "activations"), "facebook/opt-125m")
            quant_name = model_name + "-q4"
            quantize(model_name, quant_name, "facebook/opt-125m")
            eval(quant_name, quant_name.replace("models", "activations"), "facebook/opt-125m", quantized=True)

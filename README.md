# Quantization-Awareness


## Download the data

```bash
mkdir data -p
cd data
git lfs clone git@github.com:anthropics/hh-rlhf.git
cd hh-rlhf
rm -rf helpful-online helpful-rejection-sampled red-team-attempts
cd harmless-base
gzip -d train.jsonl.gz
gzip -d test.jsonl.gz
cd ../helpful-base
gzip -d train.jsonl.gz
gzip -d test.jsonl.gz
cd ../../..
python qaware/dataset_gen.py
```

## Train & eval models

```bash
python qaware/finetune.py ft facebook/opt-125m models/opt-125m facebook/opt-125m --epochs 10 
python qaware/eval.py models/opt-125m activations/opt-125m facebook/opt-125m
python qaware/quantize.py models/opt-125m models/opt-125m-q4 facebook/opt-125m
python qaware/eval.py models/opt-125m-q4 activations/opt-125m-q4 facebook/opt-125m --quantized
```
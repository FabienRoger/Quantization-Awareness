# Quantization-Awareness


## Download the data

```bash
mkdir data/hh -p
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
## Evaluating the Robustness and Accuracy of Text Watermarking Under Real-World Cross-Lingual Manipulations
In this repository, we evaluate four watermarking methods on cross-lingual translation and paraphrasing attacks. Large part in this repository was adapted from "[MarkLLM](https://github.com/THU-BPM/MarkLLM)".

## Environment Setup
1. Requirements:   <br/>
transformers <br/>
Python  <br/>
PyTorch  <br/>
openai  <br/>

2. Denpencencies:
```
pip install pandas
pip install sacrebleu
pip install sacremoses
pip install scikit-learn
pip install seaborn
pip install tqdm
```

## Experiments
[all-experiments.py](experiments/all-experiments.py) contains all necessary codes to run the experiments. <br/>
Use this file through scripts in ```experiments/scripts/*.sh```. <br/>
1. Use ```experiments/scripts/experiment-original.sh``` to generate watermarked text.
2. Use ```experiments/scripts/experiment-quality.sh``` to generate PPL and GPT-Judge related data.
3. Use ```experiments/scripts/experiment-attack.sh``` to generate our attack pipeline.
4. To reproduce the figures and tables use notebooks in ```notebooks``` folder.


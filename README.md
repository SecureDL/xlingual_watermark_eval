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

# 🌐 Multilingual Watermarking Leaderboard (**For Quality Experiments, Including LLM-as-a-Judge Scores**)

This leaderboard summarizes the performance of watermarking methods across languages and models, using key metrics: Detection Rate (TPR@FPR), Adjusted Diversity (AD), Self-BLEU (SB), Relative Decrease (RD), and LLM-as-a-Judge Softwins. Lower AD, SB, and RD indicate better quality; higher TPR at low FPR and higher Softwins mean stronger detection and better text quality. All metrics are considered for ranking, with text quality prioritized.

## 🏆 Overall Best Watermark Method (Average Across All Languages & Tasks)

| Method   | Avg TPR@FPR=0.1% | Avg TPR@FPR=1% | Avg AD | Avg SB | Avg RD | Avg Softwins | 🥇 Rank |
|----------|------------------|----------------|--------|--------|--------|--------------|---------|
| KGW      | 0.93             | 0.98           | **0.32** | **0.15** | **5.2**  | **0.50**     | 1       |
| EXP      | **0.98**         | **0.99**       | 0.36   | 0.19   | 18.5   | 0.36         | 2       |
| Unigram  | 0.67             | 0.93           | 0.39   | 0.13   | 8.2    | 0.45         | 3       |
| XSIR     | 0.96             | 0.98           | 0.41   | 0.16   | 21.2   | 0.32         | 4       |

*KGW achieves the best balance between detection and text quality (AD, SB, RD, Softwins). EXP has slightly higher detection but lower text quality.*

## 🏅 Best Method Per Language (C4 Task, Cohere Model)

| Language    | Best Method | TPR@FPR=0.1% | AD   | SB   | RD    | Softwins |
|-------------|-------------|--------------|------|------|-------|----------|
| English     | KGW         | 0.992        | 0.27 | 0.16 | -1.76 | 0.546    |
| Arabic      | KGW         | 0.998        | 0.30 | 0.11 | 2.24  | 0.33     |
| Chinese     | KGW         | 0.628        | 0.19 | 0.04 | 5.67  | 0.582    |
| Indonesian  | KGW         | 0.976        | 0.29 | 0.10 | 7.48  | 0.458    |
| Turkish     | KGW         | 0.972        | 0.25 | 0.19 | 0.20  | 0.20     |
| Hindi       | KGW         | 0.690        | 0.22 | 0.04 | 0.04  | 0.04     |

*KGW is the best method per language when considering all metrics.*

## 🥇 Best Method Per Task (Cohere Model)

| Task  | Best Method | Avg TPR@FPR=0.1% | Avg AD | Avg SB | Avg RD | Avg Softwins |
|-------|-------------|------------------|--------|--------|--------|--------------|
| LFQA  | KGW         | 0.95             | 0.27   | 0.22   | 0.87   | 0.488        |
| C4    | KGW         | 0.93             | 0.27   | 0.21   | -1.76  | 0.546        |

*KGW preserves text quality best for both tasks, with strong detection and highest Softwins.*

## 🏆 Best Language for Each Method (C4 Task, Cohere Model)

| Method   | Best Language | TPR@FPR=0.1% | AD   | SB   | RD    | Softwins |
|----------|---------------|--------------|------|------|-------|----------|
| KGW      | English       | 0.992        | 0.27 | 0.16 | -1.76 | 0.546    |
| Unigram  | English       | 0.994        | 0.27 | 0.17 | 3.18  | 0.47     |
| XSIR     | Chinese       | 0.906        | 0.23 | 0.03 | 27.81 | 0.372    |
| EXP      | English       | 1.000        | 0.27 | 0.19 | 21.4  | 0.282    |

## How to Read This Leaderboard

- **TPR@FPR=0.1%**: True Positive Rate at 0.1% False Positive Rate (higher is better).
- **AD (Adjusted Diversity)**: Lower means more diverse outputs.
- **SB (Self-BLEU)**: Lower means less repetitive outputs.
- **RD (Relative Decrease)**: Lower means less drop in GPT-judger scores (quality preserved) between watermarked and unwatermared texts.
- **Softwins (LLM-as-a-Judge)**: Higher means better judged watermarked text quality.

> **Conclusion:**  
> KGW is the best watermark method overall when balancing detection and text quality, including LLM-as-a-Judge scores. EXP is a strong alternative if detection is the only priority, but KGW preserves output quality much better across languages and tasks.

---

# 🌐 Multilingual Fairness Leaderboard (**Language Bias in LLM-as-a-Judge Experiments**)

This leaderboard ranks languages by how strongly they are affected by LLM-as-a-Judge bias, based on translation and paraphrase experiments. Higher bias means the LLM shows stronger preference or aversion, indicating less fairness.

## Language Bias Leaderboard

| Rank | Language | Translation Preference (%) | Paraphrase TIE (%) | Paraphrase Perturbed (%) | Paraphrase Natural (%) | Bias Summary |
|------|----------|--------------------------|--------------------|-------------------------|-----------------------|--------------|
| 1    | FA (Persian) | 2.00 ± 0.00           | 40.5 ± 0.71        | 39.5 ± 3.54             | 20.0 ± 4.24           | Strong bias against Persian; lowest translation win rate, high TIE rate, low natural text preference. |
| 2    | ZH (Chinese) | 7.00 ± 1.41           | 44.0 ± 1.41        | 30.0 ± 1.41             | 26.0 ± 0.0            | Strong bias against Chinese; low translation win rate, highest TIE rate, low natural text preference. |
| 3    | ID (Indonesian) | 3.50 ± 0.71        | 21.5 ± 3.54        | 48.0 ± 4.24             | 30.5 ± 7.78           | Bias against Indonesian; very low translation win rate, moderate TIE, high perturbed text preference. |
| 4    | JA (Japanese) | 14.50 ± 3.54         | 37.5 ± 6.36        | 34.0 ± 7.07             | 28.5 ± 0.71           | Moderate bias; low translation win rate, high TIE, moderate natural text preference. |
| 5    | EN (English) | 19.50 ± 4.95          | 0.5 ± 0.71         | 75.5 ± 9.19             | 24.0 ± 9.90           | Mild bias; moderate translation win rate, very low TIE, strong preference for perturbed text. |
| 6    | AR (Arabic) | 23.00 ± 5.66           | 29.0 ± 1.41        | 40.5 ± 4.95             | 30.5 ± 6.36           | Mild bias; high translation win rate, moderate TIE, balanced perturbed/natural preference. |
| 7    | DE (German) | 30.50 ± 3.54           | 23.0 ± 8.49        | 47.5 ± 6.36             | 28.5 ± 2.12           | Least affected; highest translation win rate, moderate TIE, balanced perturbed/natural preference. |

**Interpretation:**  
- Persian, Chinese, and Indonesian are most affected by LLM-as-a-Judge bias, with low translation win rates and high TIE rates in paraphrase experiments.
- German and Arabic are least affected, with higher translation win rates and more balanced paraphrase outcomes.
- English and Japanese show moderate bias, with English favoring perturbed texts and Japanese showing higher TIE rates.

> **Conclusion:**  
> LLM-as-a-Judge exhibits significant language bias, especially against Persian, Chinese, and Indonesian. This should be considered when evaluating multilingual text quality and fairness.

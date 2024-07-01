## [ACL 24 (main)] TG-LLM: Large Language Models Can Learn Temporal Reasoning

This repository contains the code for the paper [Large Language Models Can Learn Temporal Reasoning](https://arxiv.org/pdf/2401.06853.pdf).

Our framework (TG-LLM) performs temporal reasoning in two steps: 1) Text-to-Temporal Graph translation: generate (relevant) temporal graph given the context and keyword (extracted from questions); 2) Temporal Graph Reasoning: perform Chain-of-Thought reasoning over the temporal graph.

<br>

<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/TG-LLM/main/misc/Framework.png' width=550>
</p>




## Quick Start

We use [Hugging Face](https://huggingface.co/) platform to load the Llama2 model family. Make sure you have an account ([Guidance](https://huggingface.co/blog/llama2)).

The structure of the file folder should be like
```sh
TG-LLM/
│
├── materials/
│
├── model_weights/
│
├── results/
│
└── src/
```

Preparation
```sh
# git clone this repo

# create a new environment with anaconda and install the necessary Python packages

# install hugging face packages to load Llama2 models and datasets

# create the folders
cd TG-LLM
mkdir model_weights
mkdir results
```

For our TG-LLM framework

```sh
cd src

# step 1: text-to-temporal graph translation
python SFT_with_LoRA_text_to_TG_Trans.py

# step 2: temporal graph reasoning
python CoT_bootstrap.py
python SFT_with_LoRA_TG_Reasoning.py

# to obtain results based on perplexity
python SFT_with_LoRA_TG_Reasoning_ppl.py
```

For other leading LLMs (GPT series/Llama2 family)
```sh
cd src
python Inference_in_context_learning.py

# to obtain results based on perplexity
python Inference_in_context_learning_ppl.py
```

For evaluation
```sh
cd src
python Evaluation.py
```


## Datasets

All the datasets (TGQA, TimeQA, TempReason) can be found [here](https://huggingface.co/datasets/sxiong/TGQA).

To download the dataset, install [Huggingface Datasets](https://huggingface.co/docs/datasets/quickstart) and then use the following command:

```python
from datasets import load_dataset
dataset = load_dataset("sxiong/TGQA", "TGQA")
dataset = load_dataset("sxiong/TGQA", "TimeQA")
dataset = load_dataset("sxiong/TGQA", "TempReason")
```

## Contact
If you have any inquiries, please feel free to raise an issue or reach out to sxiong45@gatech.edu.

## Citation
```
@misc{xiong2024large,
      title={Large Language Models Can Learn Temporal Reasoning}, 
      author={Siheng Xiong and Ali Payani and Ramana Kompella and Faramarz Fekri},
      year={2024},
      eprint={2401.06853},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Visual Token Compression for Efficient Multimodal LLMs

Research code for studying visual token compression in multimodal large language models (MLLMs), with a focus on improving computational efficiency while preserving visual-language performance.

This repository is based on the publicly released VisionSelector codebase and is used as a research starting point for experimentation, adaptation, and evaluation of visual token compression methods.

Overview

Multimodal large language models can process detailed visual information by converting images or videos into sequences of visual tokens. However, these token sequences can become computationally expensive, particularly when high-resolution images or long videos are used.

Visual token compression addresses this problem by reducing the number of visual tokens passed to the language model while attempting to preserve the information most relevant to the downstream task.

This repository explores this problem through experiments with learnable and non-learnable token selection strategies.

The main research direction is:

How can visual token sequences be compressed efficiently without substantially degrading multimodal understanding?

The codebase currently supports experimentation with visual token compression in multimodal models including Qwen2.5-VL and LLaVA-OneVision.

Research Focus

The experiments in this repository investigate:

Visual token selection and pruning
Token compression for multimodal LLMs
Efficient multimodal inference
Retention-rate vs. performance trade-offs
Computational and memory efficiency
Learnable importance-based token selection
Comparison between different token compression strategies
Evaluation of compressed multimodal models

The broader goal is to understand what visual information can be removed and what information should be preserved when reducing the computational cost of MLLMs.

Relationship to VisionSelector

This repository builds upon the publicly available implementation of:

VisionSelector: End-to-End Learnable Visual Token Compression for Efficient Multimodal LLMs

The original project introduced an end-to-end learnable framework for visual token compression, including a differentiable Top-K selection mechanism, curriculum-based training, and a learnable importance scorer.

The original project is used here as a research baseline and implementation starting point.

This repository should therefore be understood as an experimental/extended codebase rather than a reimplementation claiming ownership of the original VisionSelector method.

For the original method, results, and publication, please refer to the original VisionSelector project and paper.

Repository Structure
.
├── datasets/
│   └── Dataset preparation and annotation utilities
│
├── docs/
│   └── Documentation and figures
│
├── qwen-vl-finetune/
│   └── Qwen2.5-VL training and token-selection experiments
│
├── qwen-evaluation/
│   └── Evaluation and inference scripts
│
├── qwen-vl-utils/
│   └── Supporting utilities
│
├── llava-ov-15/
│   └── LLaVA-OneVision-1.5 experiments
│
├── lmms-eval/
│   └── Multimodal benchmark evaluation
│
├── requirements.txt
└── README.md
Dataset Preparation

The original experimental pipeline uses datasets from the Cambrian-10M dataset collection.

The required datasets include:

Dataset	Approx. Size
OCR-VQA	~80K
ChartQA	~28K
TextVQA	~21K
COCO	~364K

The corresponding annotation file used by the original pipeline is:

Cambrian737k.jsonl

These datasets are not included in this repository and should be obtained from their respective public sources.

Dataset Preparation

After downloading the required datasets, place them inside:

datasets/

The expected structure is approximately:

datasets/
├── ocr_vqa/
├── ocr_vqa_cambrian.jsonl
├── chartqa/
├── chartqa_cambrian.jsonl
├── textvqa/
├── textvqa_cambrian.jsonl
├── coco/
├── coco_cambrian.jsonl
└── textvqa_ocrvqa_cambrian.jsonl

The annotation-processing utilities can then be used to prepare the required JSONL files.

python datasets/filter_json.py
python datasets/sample_merge_json_llavaov.py

The exact commands and paths may need to be adjusted according to the local dataset configuration.

Environment Setup

The experiments were developed around a Python/Conda environment.

Create an environment with:

conda create -n vision-compression python=3.10
conda activate vision-compression

Install the project dependencies:

pip install -r requirements.txt

For the Qwen2.5-VL pipeline:

pip install qwen-vl-utils[decord]
pip install transformers==4.50.0

For the LLaVA-OneVision pipeline, the required Transformers version may differ from the Qwen environment.

pip uninstall transformers
pip install transformers==4.53.1
Qwen2.5-VL Experiments

The repository contains an experimental pipeline for studying visual token compression with Qwen2.5-VL.

Training

Training scripts are available under:

qwen-vl-finetune/

For example:

cd qwen-vl-finetune

The available scripts include configurations for different experimental settings:

bash scripts/sft_7b.sh
bash scripts/sft_3b.sh
bash scripts/sft_dynamic.sh

These scripts correspond to the configurations inherited from the underlying research codebase and may be modified for further experiments.

Evaluation

Evaluation utilities are provided through the lmms-eval and qwen-evaluation components.

Install the evaluation package:

cd lmms-eval
pip install -e .
cd ../qwen-evaluation

Evaluation scripts can be used to compare:

Original model inference
Token compression baselines
Learnable token selection
Other supported token pruning approaches

Example:

bash run_token_compression.sh

For selector-based evaluation:

bash run_selector.sh

For dynamic token selection experiments:

bash run_dynamic_qwen.sh
Inference

Inference experiments can be run using:

bash run_inference.sh

The pipeline supports experiments involving different visual token retention strategies.

The goal is to measure how reducing the number of visual tokens affects:

Multimodal task performance
Number of visual tokens
Inference latency
Prefill time
GPU memory consumption
Efficiency Evaluation

For efficiency experiments, the evaluation pipeline can record:

Maximum GPU Memory
Prefill Time
Latency
Number of Visual Tokens

Enable timing and resource measurements with:

EVAL_TIME=True

These measurements can be used to analyze the trade-off between visual token retention and computational efficiency.

LLaVA-OneVision Experiments

The repository also contains an experimental implementation for LLaVA-OneVision-1.5 under:

llava-ov-15/

Install the required environment dependencies and configure the appropriate Transformers version before running the experiments.

Training
cd llava-ov-15
bash scripts/finetune_selector_8b.sh
Evaluation
bash run_ov_token_compression.sh

Selector-based evaluation:

bash run_ov_selector.sh
Inference
bash run_ov_inference.sh
Experimental Direction

The codebase is intended to support further investigation into efficient multimodal models.

Potential experimental directions include:

1. Token Retention

Study model performance under different visual token budgets:

100% → 75% → 50% → 25% → 10%
2. Selection Strategies

Compare different approaches for determining which visual tokens should be retained.

3. Efficiency

Measure the relationship between token reduction and:

GPU memory
Latency
Prefill time
Throughput
4. Generalization

Investigate whether a token-selection strategy trained under one compression setting remains effective when evaluated at different retention rates.

5. Model Comparison

Evaluate token compression across different multimodal architectures rather than restricting experiments to a single backbone.

Current Status

This repository is a research and experimentation codebase.

New experiments, modifications, and findings will be documented as the research progresses.
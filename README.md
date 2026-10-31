# PyTorch-Based Code Summarization Tool for Python

## Overview

This project implements an automatic code summarization system using deep learning.
The system generates short natural-language summaries (docstrings) from Python source
code to improve code understanding, readability, and documentation.

The solution is based on a Sequence-to-Sequence (Seq2Seq) Encoder–Decoder architecture
with an attention mechanism, implemented in PyTorch.

## Features

- Automatic code-to-text summarization
- Seq2Seq (Encoder–Decoder) model with attention
- Custom tokenization and preprocessing
- Training and evaluation pipeline
- BLEU and ROUGE evaluation metrics
- Inference module for summarizing unseen code

## Project Structure

mlsaSeq2SeqProject/
│
├── data/
│   └── sample_data2.csv
│
├── models/
│   └── model.pth
│
├── src/
│   ├── tokenizer.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── summarize.py
│
└── README.md

## Requirements

- Python 3.11
- PyTorch
- Pandas
- NumPy
- NLTK
- rouge-score

## Setup Instructions

1. Open the project directory
   cd D:\mlsaSeq2SeqProject

2. Create a virtual environment
   py -3.11 -m venv venv

3. Activate the virtual environment (Windows CMD)
   venv\Scripts\activate

4. Install dependencies
   python -m pip install torch pandas numpy nltk rouge-score

## Execution Steps

### Train the Model

python src/train.py

This script trains the Seq2Seq model and saves the trained model to models/model.pth.

### Evaluate the Model

python src/evaluate.py

This script evaluates the trained model using BLEU and ROUGE metrics.

### Run Inference

python src/summarize.py

Example input:
def sum_two(a, b): return a + b

Example output:
returns a + b

## Dataset

The dataset consists of Python code snippets paired with natural-language docstrings.
It is inspired by the CodeSearchNet dataset and stored in CSV format.

## Model Architecture

- Seq2Seq Encoder–Decoder (LSTM-based)
- Attention mechanism
- Cross Entropy Loss
- Adam Optimizer

## Evaluation Metrics

- Cross Entropy Loss
- BLEU Score
- ROUGE-1 and ROUGE-L

## Authors

- Muhammad Mubeen Shar Gil
- Kumar Laveet
- Ahmed Masood

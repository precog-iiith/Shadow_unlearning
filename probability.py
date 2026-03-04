import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import datasets, joblib
import json
from datasets import load_from_disk

model_path = "" 
tokenizer_path = "" 
dataset_path = ''

model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",                 
        torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token



def load_dataset_from_jsonl(file_path: str) -> list[dict]:
    """Loads a dataset from a JSONL file, preserving all fields."""
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                samples.append(json.loads(line))
        return samples
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading dataset from {file_path}: {e}")
        return []


dataset = {"question":list(pd.read_json(dataset_path, lines = True).question),
    "answer":list(pd.read_json(dataset_path, lines = True).answer)}
dataset = datasets.Dataset.from_dict(dataset)


def _calculate_conditional_log_prob(model, data):
        model.eval()
        total_nll = 0
        total_samples = len(data)

        if total_samples == 0:
            return float('inf')

        with torch.no_grad():
            for item in tqdm(data, desc="Calculating Conditional Log-Prob", leave=False):
                question = item['question']
                answer = item['answer']
                question_tokens = tokenizer(question, return_tensors='pt').input_ids.to(model.device)
                answer_tokens = tokenizer(answer, return_tensors='pt').input_ids.to(model.device)
                input_ids = torch.cat([question_tokens, answer_tokens], dim=1)
                labels = input_ids.clone()
                labels[:, :question_tokens.shape[1]] = -100
                outputs = model(input_ids=input_ids, labels=labels)
                total_nll += outputs.loss.item()
        print(f"Conditional Log-Prob: {total_nll / total_samples}")
        return total_nll / total_samples

_calculate_conditional_log_prob(model, forget_dataset)
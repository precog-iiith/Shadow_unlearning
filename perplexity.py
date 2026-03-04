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


def _calculate_perplexity_batched(model, data):
        model.eval()
        total_neg_log_likelihood = 0
        total_tokens = 0
        if not data:
            return float('inf')
    
        with torch.no_grad():
            for i in tqdm(range(0, len(data), 8), desc="Calculating Perplexity", leave=False):
                batch_data = retain_dataset[i:i+8]
                batch_texts = [q + " " + a for q, a in zip(batch_data['question'], batch_data['answer'])]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(model.device)
    
                labels = inputs.input_ids.clone()
                labels[inputs.attention_mask == 0] = -100  
                outputs = model(**inputs, labels=labels)
    
                num_tokens_in_batch = inputs.attention_mask.sum().item()
                total_neg_log_likelihood += outputs.loss.item() * num_tokens_in_batch
                total_tokens += num_tokens_in_batch
    
        if total_tokens == 0:
            return float('inf')
    
        avg_neg_log_likelihood = total_neg_log_likelihood / total_tokens
        print(f"Perplexity: {np.exp(avg_neg_log_likelihood)}")
        return np.exp(avg_neg_log_likelihood)

_calculate_perplexity_batched(model, dataset)
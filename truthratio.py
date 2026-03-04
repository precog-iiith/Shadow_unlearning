import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_log_prob(model, tokenizer, prompt, target):
    full_text = prompt + target
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, prompt_len-1:-1, :]
        target_ids = inputs["input_ids"][:, prompt_len:]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        return -loss.mean().item() 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="JSONL file with question, answer, and perturbed_answers")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    df = pd.read_json(args.data_path, lines=True)
    ratios = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        true_ans = row['answer']
        perturbed_ans_list = row['perturbed_answer'] # Expecting a list of strings

        lp_true = compute_log_prob(model, tokenizer, question, true_ans)
        prob_true = torch.exp(torch.tensor(lp_true))

        lp_perturbed = [compute_log_prob(model, tokenizer, question, p) for p in perturbed_ans_list]
        avg_prob_perturbed = torch.exp(torch.tensor(lp_perturbed)).mean()

        ratio = (avg_prob_perturbed / prob_true).item()
        ratios.append(ratio)

    print(f"\nFinal Averaged Truth Ratio: {sum(ratios)/len(ratios):.4f}")

if __name__ == "__main__":
    main()
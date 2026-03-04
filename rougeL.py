import pandas as pd
import argparse
from tqdm import tqdm
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def get_first_sentence(text):
    if not text or not isinstance(text, str):
        return ""
    sentences = sent_tokenize(text.strip())
    return sentences[0] if sentences else ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", type=str, required=True, help="Path to gold JSONL")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to model predictions")
    parser.add_argument("--pred_column", type=str, default="unlearned_response", help="Column name for predictions")
    args = parser.parse_args()

    gold_df = pd.read_json(args.gold_path, lines=True)
    pred_df = pd.read_json(args.pred_path, lines=True if 'jsonl' in args.pred_path else False)

    gold_answers = gold_df['answer'].tolist()
    model_preds = pred_df[args.pred_column].tolist()

    scores = []
    for gold, pred in tqdm(zip(gold_answers, model_preds), total=len(gold_answers)):
        clean_pred = get_first_sentence(pred)
        score = scorer.score(gold, clean_pred)['rougeL'].fmeasure
        scores.append(score)

    avg_rouge_l = sum(scores) / len(scores) if scores else 0
    print(f"\n--- Results ---")
    print(f"Samples Evaluated: {len(scores)}")
    print(f"Average ROUGE-L:   {avg_rouge_l:.4f}")

if __name__ == "__main__":
    main()
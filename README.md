This repository contains the datasets and code for the paper "Shadow Unlearning: A Neuro-Semantic Approach to Fidelity-Preserving Faceless Forgetting in LLMs". 

## File Structure

### Evaluation Scripts
The following are the metric-wise evaluation scripts. Once the results for all the model-data combinations are obtained, the reported metrics of HPS, HRS, CES, and HCNLL can be computed as described in the paper.
- **`rougeL.py`**: Computes ROUGE-L scores to measure textual similarity between model outputs and gold answers.
- **`truthratio.py`**: Calculates the Truth Ratio (Likelihood of Perturbed Answers / Likelihood of Gold Answer).
- **`probability.py`**: Computes conditional log-probabilities and NLL for specific QA pairs.
- **`perplexity.py`**: Measures model fluency and next-token prediction certainty.

### Datasets
- **`forget_datasets/`**: Raw datasets containing the information intended for unlearning.
- **`anonymized_forget_datasets/`**: Forget sets with sensitive entities or identifiers masked.
- **`all_retain_dataset/`**: Information that is supposed to be retained.

### Pipeline
- **`NSPU_pipeline.ipynb`**: The primary execution notebook for Neuro-semantic projection unlearning.

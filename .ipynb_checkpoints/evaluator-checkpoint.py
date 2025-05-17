import torch
from detoxify import Detoxify
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

class Evaluator:
    """
    Evaluator for measuring toxicity and semantic similarity of model outputs.

    Args:
        model: a Hugging Face Seq2Seq model (e.g., T5ForConditionalGeneration).
        tokenizer: corresponding tokenizer.
        device: torch.device to run evaluation on (defaults to GPU if available).
    """
    def __init__(self, model, tokenizer, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # Toxicity classifier from Detoxify
        self.toxicity_model = Detoxify('original')
        # Sentence-transformers model for embeddings
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

    def evaluate(self, dataset, batch_size=8, max_length=64):
        """
        Evaluate the model on a dataset split.

        Args:
            dataset: an iterable of dicts with keys 'toxic' and 'neutral'.
            batch_size: number of examples per generation batch.
            max_length: maximum generation length.

        Returns:
            dict with:
                avg_toxicity: mean toxicity score of generated outputs.
                avg_similarity: mean cosine similarity between outputs and references.
                tox_scores: list of per-example toxicity scores.
                similarities: list of per-example cosine similarities.
        """
        # Prepare inputs and references
        inputs = [f"detoxify: {ex['toxic']}" for ex in dataset]
        references = [ex['neutral'] for ex in dataset]

        # Generate predictions
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(inputs), batch_size), desc="Generating outputs"):
                batch_texts = inputs[i:i+batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                out_ids = self.model.generate(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    max_length=max_length
                )
                decoded = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                preds.extend(decoded)

        # Compute toxicity scores for generated texts
        tox_scores = [self.toxicity_model.predict(text)['toxicity'] for text in preds]
        avg_toxicity = float(np.mean(tox_scores))

        # Compute embeddings and cosine similarities
        refs_emb = self.sim_model.encode(references, convert_to_tensor=True)
        preds_emb = self.sim_model.encode(preds, convert_to_tensor=True)
        cos_sims = torch.nn.functional.cosine_similarity(preds_emb, refs_emb)
        avg_similarity = float(cos_sims.mean().item())

        return {
            'avg_toxicity': avg_toxicity,
            'avg_similarity': avg_similarity,
            'tox_scores': tox_scores,
            'similarities': cos_sims.tolist()
        }


if __name__ == "__main__":
    # Example usage
    import yaml
    from data_processing import load_dataset_from_disk
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    # Load config for model checkpoint
    cfg = yaml.safe_load(open('main_config.yml'))
    output_dir = cfg.get('sft_params', {}).get('output_dir', './outputs/SFT')

    # Load model and tokenizer from last checkpoint
    model = T5ForConditionalGeneration.from_pretrained(output_dir)
    tokenizer = T5Tokenizer.from_pretrained(output_dir)

    # Load evaluation split
    eval_dataset = load_dataset_from_disk('eval_dataset')

    # Initialize evaluator
    evaluator = Evaluator(model, tokenizer)

    # Run evaluation
    results = evaluator.evaluate(eval_dataset)
    print("Average Toxicity:", results['avg_toxicity'])
    print("Average Similarity:", results['avg_similarity'])

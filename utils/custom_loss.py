import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


class SimilarityToxicityReward(nn.Module):
    def __init__(self, w_sim: float = 1.0, w_tox: float = 1.0, device: str = None):
        super().__init__()
        self.cat_wts = {
                'toxic': 1,
                'severe_toxic': 0,
                'obscene': 0,
                'threat': 0,
                'insult': 0,
                'identity_hate': 0
            }
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.toxicity_model_name = "unitary/toxic-bert"
        self.similarity_model_name = "princeton-nlp/sup-simcse-bert-base-uncased"


        self.initialize_models_and_tokenizers()

        self.sim_mod.to(self.device)
        self.tox_mod.to(self.device)

        # toxicity weights
        label2id = self.tox_mod.config.label2id
        w = torch.zeros(len(label2id), dtype=torch.float)
        for lab, wt in self.cat_wts.items():
            w[label2id[lab]] = wt
        w = w / w.sum()
        self.register_buffer("tox_weights", w)

        self.w_sim = w_sim
        self.w_tox = w_tox
    def initialize_models_and_tokenizers(self):
        # Load the models and tokenizers
        self.tox_tok = AutoTokenizer.from_pretrained(self.toxicity_model_name)
        self.tox_mod = AutoModelForSequenceClassification.from_pretrained(self.toxicity_model_name).eval()
        self.sim_tok = AutoTokenizer.from_pretrained(self.similarity_model_name)
        self.sim_mod = AutoModel.from_pretrained(self.similarity_model_name).eval()
         
    def forward(self, orig_sentences, new_sentences):
        # ensure lists
        if isinstance(orig_sentences, str):
            orig_sentences = [orig_sentences]
            new_sentences = [new_sentences]
        B = len(orig_sentences)

        # 1) similarity loss
        sim_inputs = self.sim_tok(
            orig_sentences + new_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        emb = self.sim_mod(**sim_inputs, return_dict=True).pooler_output  # (2B, hidden)
        orig_emb, new_emb = emb.split(B, dim=0)
        cos_sim = F.cosine_similarity(orig_emb, new_emb, dim=1)          # (B,)
        reward_sim = cos_sim                                         # (B,)

        # 2) toxicity loss
        tox_inputs = self.tox_tok(
            new_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        logits = self.tox_mod(**tox_inputs).logits                       # (B, num_labels)
        probs = torch.sigmoid(logits)
        tox_weights = self.tox_weights.to(probs.device)                             # (B, num_labels)
        weighted = (probs * tox_weights.unsqueeze(0)).sum(dim=1)    # (B,)
        reward_tox = 1- weighted                                             # (B,)

        # combined per-example
        return self.w_sim * reward_sim + self.w_tox * reward_tox            # (B,)
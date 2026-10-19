"""
Load the fine-tuned ElementBERT model and evaluate semantic similarity
between ANSYS finite element types using cosine similarity of their
embeddings, visualized as a heatmap.
"""

import logging
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "./elementbert-mlm"
ELEMENT_LIST_FILE = "ansys_element_types.txt"
OUTPUT_HEATMAP = "element_similarity_heatmap.png"
NUM_SAMPLES = 10
RANDOM_SEED = 67


def load_element_list(path: str = ELEMENT_LIST_FILE) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_embedding(text: str, tokenizer, model, device) -> np.ndarray:
    """Compute a sentence-level embedding via mean pooling over token embeddings."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling over the last hidden state, masking out padding tokens
    last_hidden_state = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)

    summed = torch.sum(last_hidden_state * attention_mask, dim=1)
    counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / counts

    return mean_pooled.squeeze(0).cpu().numpy()


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-9, None)
    return normalized @ normalized.T


def plot_heatmap(similarity_matrix: np.ndarray, labels: list[str], out_path: str) -> None:
    plt.figure(figsize=(max(8, len(labels) * 0.4), max(6, len(labels) * 0.4)))
    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Cosine similarity"},
    )
    plt.title("ElementBERT: Pairwise Element Type Similarity")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    logger.info("Saved heatmap to %s", out_path)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading model and tokenizer from %s", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    element_types = load_element_list()
    logger.info("Loaded %d element types", len(element_types))

    random.seed(RANDOM_SEED)
    sample_size = min(NUM_SAMPLES, len(element_types))
    element_types = random.sample(element_types, sample_size)
    logger.info("Randomly sampled %d element types: %s", sample_size, element_types)

    logger.info("Computing embeddings...")
    embeddings = np.stack([
        get_embedding(el, tokenizer, model, device) for el in element_types
    ])

    logger.info("Computing cosine similarity matrix...")
    sim_matrix = cosine_similarity_matrix(embeddings)

    plot_heatmap(sim_matrix, element_types, OUTPUT_HEATMAP)


if __name__ == "__main__":
    main()
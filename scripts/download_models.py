from pathlib import Path

from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SAVE_PATH = Path(__file__).parent.parent / "models" / "sapbert"

print(f"Downloading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

SAVE_PATH.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"Saved to {SAVE_PATH.resolve()}")

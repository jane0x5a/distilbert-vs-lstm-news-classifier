import os
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "data")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# Load test data
dataset = load_dataset(
    "csv",
    data_files={"test": os.path.join(DATA_DIR, "ag_news_test.csv")},
    delimiter=",",
)
test_texts = list(dataset["test"]["text"])
test_labels = list(dataset["test"]["label"])

# Tokenize
encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
input_ids = torch.tensor(encodings["input_ids"])
attention_mask = torch.tensor(encodings["attention_mask"])
labels = torch.tensor(test_labels)

# Prepare DataLoader
test_dataset = TensorDataset(input_ids, attention_mask, labels)
test_loader = DataLoader(test_dataset, batch_size=8)

# Evaluate
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

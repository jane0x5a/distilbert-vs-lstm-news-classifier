import os
import time
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from app.utils import timing

# setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "data")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "distilbert-base-uncased"
epochs = 5

# process data
dataset = load_dataset(
    "csv",
    data_files={
        "train": os.path.join(DATA_DIR, "ag_news_train.csv"),
        "test": os.path.join(DATA_DIR, "ag_news_test.csv")
    },
    delimiter=",",
)
train_dataset = dataset["train"].select(range(5000))
test_dataset = dataset["test"].select(range(500))
train_texts = list(train_dataset["text"])
train_labels = list(train_dataset["label"])

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
input_ids = torch.tensor(train_encodings["input_ids"])
attention_mask = torch.tensor(train_encodings["attention_mask"])
labels = torch.tensor(train_labels)

# Dataset + Loader
dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# model
distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=4)
distilbert_model.to(device)

# only train the last 2 layers
for param in distilbert_model.distilbert.parameters():
    param.requires_grad = False
for layer in distilbert_model.distilbert.transformer.layer[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

# optimizer
optimizer = AdamW(distilbert_model.parameters(), lr=5e-5)

# training
print("Start training DistilBERT...")
distilbert_model.train()
start_time = time.time()
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = distilbert_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

print(f"Training finished in {timing(time.time() - start_time)}")

# save model
os.makedirs(MODEL_DIR, exist_ok=True)
distilbert_model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("DistilBERT model saved!")
import torch
import pandas as pd
import os
import pickle
from torch.utils.data import TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from app.LSTMclassifier import LSTMClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "data")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")

    train = pd.read_csv(os.path.join(data_dir, "ag_news_train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "ag_news_test.csv"))

    return train, test


def process_lstm_train(train_df, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train_df["text"])
    y_train = torch.tensor(train_df["label"].values, dtype=torch.long)
    return vectorizer, x_train, y_train


def process_lstm_test(test_df, vectorizer):
    x_test = vectorizer.transform(test_df["text"])
    y_test = torch.tensor(test_df["label"].values, dtype=torch.long)
    return x_test, y_test


def get_lstm_model(input_dim, hidden_dim=256, output_dim=4):
    return LSTMClassifier(input_dim, hidden_dim, output_dim)


def load_lstm_model(model_path="models/lstm_model.pt", vectorizer_path="models/vectorizer.pkl"):
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    input_dim = vectorizer.transform(["dummy"]).shape[1]
    model = LSTMClassifier(input_dim=input_dim, hidden_dim=256, output_dim=4)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, vectorizer


def process_bert_model(train_df):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    encoded = tokenizer(
        list(train_df["text"]),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    labels = torch.tensor(train_df["label"].values, dtype=torch.long)
    return encoded, labels, tokenizer


def get_bert_model(num_labels=4, device=None):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )
    if device:
        model.to(device)
    model.eval()
    return model


def load_distilbert_model():
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    return model, tokenizer

import torch
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report

label2class = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Tech"
}

class2label = {v: k for k, v in label2class.items()}


def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(label2class.values()))
    return acc, report


def logits_to_preds(logits):
    return torch.argmax(logits, dim=1).cpu().numpy()


def clean_text(text):
    return text.lower().strip()


def timing(func):
    def wrapper(*args,**kwargs):
        start_time=time.time()
        result=func(*args,**kwargs)
        print(f"{func.__name__} executed in {time.time() - start_time:.2f} seconds")
        return result
    return wrapper
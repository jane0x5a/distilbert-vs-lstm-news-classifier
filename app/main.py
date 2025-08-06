from fastapi import FastAPI
from pydantic import BaseModel
from app.model_loader import load_distilbert_model
import torch

app = FastAPI()

# loading model and tokenizer
model, tokenizer = load_distilbert_model()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# define input format
class InputText(BaseModel):
    text: str


# define label mapping（0,1,2,3 -> class）
label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


@app.get("/")
def root():
    return {"message": "Welcome to the AG News Classification API!"}


@app.get("/info")
def info():
    return {
        "model": "DistilBERT (fine-tuned)",
        "task": "AG News Classification",
        "labels": label_map
    }


@app.post("/predict")
def predict(input_text: InputText):
    # encode
    encodings = tokenizer(
        input_text.text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # inference
    with torch.no_grad():
        outputs = model(**encodings)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return {"label": label_map[predicted_class]}

import os
import torch
import torch.nn as nn
import pickle
from app.model_loader import load_data, process_lstm_train, process_lstm_test, get_lstm_model
from sklearn.metrics import accuracy_score

# hyper parameters
EPOCHS = 20
LR = 0.001
BATCH_SIZE = 64
HIDDEN_DIM = 256
OUTPUT_DIM = 4
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
VECTOR_PATH = os.path.join(ROOT_DIR, "model", "vectorizer.pkl")



def main():
    # load data
    train_df, test_df = load_data()
    vectorizer, x_train, y_train = process_lstm_train(train_df)

    # change to tensor
    x_train_tensor = torch.tensor(x_train.toarray(), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # model setup
    input_dim = x_train_tensor.shape[1]
    model = get_lstm_model(input_dim, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # training
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # save models and vectorizer
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_model.pt"))
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print("Save model and vectorizer!")

    with open(VECTOR_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    x_test, y_test = process_lstm_test(test_df, vectorizer)
    x_test_tensor = torch.tensor(x_test.toarray(), dtype=torch.float32)

    with torch.no_grad():
        outputs = model(x_test_tensor)
        pred = torch.argmax(outputs, dim=1)
        acc = accuracy_score(y_test, pred)
        print(f"\nTest Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

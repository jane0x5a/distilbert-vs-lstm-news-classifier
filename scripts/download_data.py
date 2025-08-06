from datasets import load_dataset
import pandas as pd
import os


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs("data", exist_ok=True)

    dataset = load_dataset("ag_news")

    # change to dataframe
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # save the datasets
    train_path=os.path.join(DATA_DIR,"ag_news_train.csv")
    test_path=os.path.join(DATA_DIR,"ag_news_test.csv")

    train_df.to_csv(train_path,index=False)
    test_df.to_csv(test_path,index=False)

if __name__ == "__main__":
    main()


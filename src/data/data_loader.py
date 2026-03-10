from pathlib import Path
import pandas as pd


class DataLoader:

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)

    def load_parquet(self, filename):

        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"{filename} not found")

        df = pd.read_parquet(file_path)

        return df

    def prepare_dataframe(self, df):

        # convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # sort time series
        df = df.sort_values(["patient_id", "timestamp"])

        # reset index
        df = df.reset_index(drop=True)

        return df

    def load_train(self):

        df = self.load_parquet("train_data.parquet")

        return self.prepare_dataframe(df)

    def load_validation(self):

        df = self.load_parquet("val_data.parquet")

        return self.prepare_dataframe(df)

    def load_test(self):

        df = self.load_parquet("test_data.parquet")

        return self.prepare_dataframe(df)

    def load_all(self):

        train = self.load_train()
        val = self.load_validation()
        test = self.load_test()

        return train, val, test


if __name__ == "__main__":

    loader = DataLoader()

    train_df, val_df, test_df = loader.load_all()

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)
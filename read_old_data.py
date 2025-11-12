import pandas as pd

class ParquetToCSVConverter:
    def __init__(self, parquet_path, csv_path):
        self.parquet_path = parquet_path
        self.csv_path = csv_path

    def preview(self, n=5):
        df = pd.read_parquet(self.parquet_path)
        return df.head(n)

    def convert(self):
        df = pd.read_parquet(self.parquet_path)
        df.to_csv(self.csv_path, index=False)
        print(f"Converted successfully: {self.csv_path}")


def main():
    # Change these values to your actual filenames
    parquet_file = "processed_data/EURUSDc_processed.parquet"
    csv_file = "processed_data/test_data.csv"

    converter = ParquetToCSVConverter(parquet_file, csv_file)

    print("Preview:")
    print(converter.preview())

    converter.convert()


if __name__ == "__main__":
    main()
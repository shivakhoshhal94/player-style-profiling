from src import data_loader

def main():
    df = data_loader.load_data()
    print("\n📋 Columns in the dataset:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()

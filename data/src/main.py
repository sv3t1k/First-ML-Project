from src.loader import DataLoader
from src.preprocessor import TextPreprocessor

def main():
    loader = DataLoader("data/1429_1.csv")
    df = loader.load_and_clean()
    preprocessor = TextPreprocessor()
    print("Starting text cleaning... (this may take a minute)")
    df['cleaned_text'] = df['reviews.text'].apply(preprocessor.clean_text)
    print(df[['reviews.text', 'cleaned_text', 'label']].head())
    df.to_csv("data/processed_data.csv", index=False)

if __name__ == "__main__":
    main()
    trainer = ModelTrainer()
    X = trainer.vectorize_data(df['cleaned_text'])

    print(f"Matrix is created! Size: {X.shape}") 


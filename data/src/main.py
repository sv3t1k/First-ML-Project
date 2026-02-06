from sklearn.model_selection import train_test_split

def main():

    trainer = ModelTrainer(vectorizer) 
    X = trainer.vectorize_data(df['cleaned_text'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trainer.compare_models(X_train, X_test, y_train, y_test)
    best_model = trainer.fine_tune_logistic(X_train, y_train)
    final_preds = best_model.predict(X_test)
    print(f"Финальная точность после тюнинга: {accuracy_score(y_test, final_preds):.4f}")

if __name__ == "__main__":
    main()

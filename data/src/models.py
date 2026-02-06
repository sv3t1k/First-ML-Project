from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

class ModelTrainer:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.model = None

    def vectorize_data(self, texts):
        return self.vectorizer.fit_transform(texts)

    def compare_models(self, X_train, X_test, y_train, y_test):
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000)
        }
        print("\n--- Model Comparison ---")
        results = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            acc = accuracy_score(y_test, m.predict(X_test))
            print(f"{name} Accuracy: {acc:.4f}")
            results[name] = acc
        return results

    def fine_tune_logistic(self, X_train, y_train):
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
        self.model.fit(X_train, y_train)
        print("\n--- The final model is trained with class balance ---")
        return self.model

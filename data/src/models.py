from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class ModelTrainer:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, 
            ngram_range=(1, 2)
        )

    def vectorize_data(self, texts, is_training=True):
        if is_training:
            vectors = self.vectorizer.fit_transform(texts)
            joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
        else:
            vectors = self.vectorizer.transform(texts)

        return vectors

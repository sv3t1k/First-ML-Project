from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.best_model = None

    def compare_models(self, X_train, X_test, y_train, y_test):
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc
            print(f"{name} Accuracy: {acc:.4f}")
        
        return results

    def fine_tune_logistic(self, X_train, y_train):
        print("\nLooking for best parameters (GridSearch)...")
        
        param_grid = {
            'C': [0.1, 1, 10],       # Коэффициент регуляризации
            'penalty': ['l2']        # Тип штрафа
        }
        
        grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)
        
        print(f"Best parameters: {grid.best_params_}")
        self.best_model = grid.best_estimator_
        return self.best_model

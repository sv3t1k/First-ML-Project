import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}0-9]", "", text)
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        words = [self.lemmatizer.lemmatize(w) for w in words]
        
        return " ".join(words)

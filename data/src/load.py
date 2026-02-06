import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_and_clean(self):

        cols = ['reviews.text', 'reviews.rating']
        df = pd.read_csv(self.file_path, usecols=cols)
        
        df = df.dropna(subset=['reviews.text', 'reviews.rating'])

        df['label'] = df['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)
        
        return df[['reviews.text', 'label']]

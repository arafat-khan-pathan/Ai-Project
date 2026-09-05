import os
import re
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from src.nlp_preprocessing import preprocess_text

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')                
stopwords = set(stopwords.words('english'))

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


os.makedirs('models', exist_ok=True)

if not os.path.exists('dataset/spam.csv'):
    raise FileNotFoundError("dataset not found. Please download the dataset and place it in the 'dataset' folder.")

df = pd.read_csv('dataset/spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df = df.drop_duplicates() 
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # ham=0, spam=1

df = df.dropna()
df = df[df["message"].str.strip() != ""]
df = df.drop_duplicates()




# def preprocess_text(text):
#     text = str(text).lower()  
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove non-alphabetic characters
   
#     words = text.split()
#     cleaned_words = [word for word in words if word not in stopwords]      # remove stopwords 
#     cleaned_words = [word for word in cleaned_words if word != '']  # remove empty strings
#     cleaned_words = [stemmer.stem(word) for word in cleaned_words]  # stem words
    
#     return ' '.join(cleaned_words)



df['cleaned_message'] = df['message'].apply(preprocess_text)  
df = df.dropna()
df = df.drop_duplicates()

vectorizer = TfidfVectorizer(max_features=500000)
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)


with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
    
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n🎉 SUCCESS! 'tfidf_vectorizer.pkl' and 'random_forest.pkl' have been successfully created inside the models/ folder!")
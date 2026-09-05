import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


stop_words = set(stopwords.words("english"))  # for stopwords
stemmer = PorterStemmer()  # for stemming. find word base form


def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def stem_words(tokens):
    return [stemmer.stem(word) for word in tokens]


def preprocess_text(text):

    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_words(tokens)      # tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)

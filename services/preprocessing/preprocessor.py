
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text_and_tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    
    processed_tokens = [
        stemmer.stem(word) 
        for word in tokens 
        if word.isalpha() and word not in stop_words
    ]
    
    return processed_tokens


def preprocess_bm25(text: str) -> list[str]:
    """
    معالجة خفيفة لاستعلامات وبيانات BM25: تصغير الأحرف + حذف الترقيم + تقطيع بسيط.
    (لا إزالة توقف، لا Stemming)
    """
    if not isinstance(text, str):
        return []
    # تصغير الأحرف
    text = text.lower()
    # حذف علامات الترقيم
    text = text.translate(str.maketrans('', '', string.punctuation))
    # تقطيع على الفراغات
    tokens = text.split()
    # حذف التوكنات الفارغة
    tokens = [t for t in tokens if t.strip()]
    return tokens


def preprocess_bert(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()
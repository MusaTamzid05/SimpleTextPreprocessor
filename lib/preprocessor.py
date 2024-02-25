from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re


class TextPreprocessor:
    def __init__(self):
        self.stopwords = stopwords.words("english")
        self.special_char_regex = re.compile(r"[^a-zA-Z\s]")
        self.lem = WordNetLemmatizer()


    def process(self, text):
        text = self.special_char_regex.sub("", text)
        words = word_tokenize(text)
        processed_words = []

        for word in words:
            word = word.lower()

            if word in self.stopwords:
                continue

            if len(wordnet.synsets(word)) == 0:
                continue

            word = self.lem.lemmatize(word)
            processed_words.append(word)

        return processed_words

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
import pickle
import os
import math
import numpy as np

class TextPreprocessor:
    def __init__(self):
        pass

    def process(self, text):
        raise ValueError("Not implemented")

class SimpleTextProcessor(TextPreprocessor):

    def __init__(self):
        super().__init__()
    
    def process(self, text):
        return [word.lower() for word in word_tokenize(text)]


class TextPreprocessorV1(TextPreprocessor):
    def __init__(self):
        super().__init__()
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



class TFIDFPreprocessor:
    '''
    tf = term frequency meaning number time a term
    appears on a text
    tf = log10(number_of_time_term_appears + 1)

    idf = inverse document frequency
    idf = log10(total documents /total document the term appear in)
    '''

    def __init__(self, simple_text_processor=False):
        self.total_docs = 0
        self.g_word_count = {}

        if simple_text_processor:
            self.text_preprocessor = SimpleTextProcessor()
        else:
            self.text_preprocessor = TextPreprocessorV1()


    def process(self, corpus):
        # make sure this is same corpus you use for loading

        if self.total_docs == 0:
            print("Please init or load the corpus first!!")
            exit(1)
            

        results = []

        for text in corpus:
            words = self.text_preprocessor.process(text=text)
            word_count = self._get_word_count_from(text=text)
            current_result = []

            for word in words:
                tf = math.log10(word_count[word] + 1)
                idf = math.log10(self.total_docs / self.g_word_count[word])
                result = tf * idf
                current_result.append(result)

            results.append(current_result)

        return np.array(results)



    def init_data_from_corpus(self, corpus):
        if type(corpus) != list:
            raise ValueError("Corpus needs to be list of text(documents)")

        self.total_docs = len(corpus)

        for doc in corpus:
            word_count = self._get_word_count_from(text=doc)

            for word, _ in word_count.items():

                if word not in self.g_word_count.keys():
                    self.g_word_count[word] = 0
                self.g_word_count[word] += 1


        for word, count in self.g_word_count.items():
            print(f"{word} => {count}")

    def _get_word_count_from(self, text):
        word_count = {}
        words = self.text_preprocessor.process(text=text)

        for word in words:
            if word not in word_count.keys():
                word_count[word] = 0

            word_count[word] += 1

        return word_count

    def save(self, path):
        if path.endswith(".pickle") == False:
            path += ".pickle"

        if os.path.exists(path):
            print(f"removing old {path}")
            os.remove(path)

        with open(path, "wb") as f:
            data = {
                    "g_word_count" : self.g_word_count,
                    "total_docs" : self.total_docs
                    }

            pickle.dump(data, f)

            print(f"Data saved in {path}")

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.g_word_count = data["g_word_count"]
            self.total_docs = data["total_docs"]
            print(f"load successfull {path}")






















from lib.preprocessor import TFIDFPreprocessor
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

def load_example():
     corpus = ['The sky is blue and beautiful.',
              'Love this blue and beautiful sky!',
              'The quick brown fox jumps over the lazy dog.',
              "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
              'I love green eggs, ham, sausages and bacon!',
              'The brown fox is quick and the blue dog is lazy!',
              'The sky is very blue and the sky is very beautiful today',
              'The dog is lazy but the brown fox is quick!'
    ]

     corpus = gutenberg.raw(fileids="carroll-alice.txt")
     corpus = sent_tokenize(corpus)

     corpus_preprocessor = TFIDFPreprocessor(simple_text_processor=True)
     corpus_preprocessor.init_data_from_corpus(corpus=corpus)
     corpus_preprocessor.save(path="data")


def process_example():
     corpus = ['The sky is blue and beautiful.',
              'Love this blue and beautiful sky!',
              'The quick brown fox jumps over the lazy dog.',
              "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
              'I love green eggs, ham, sausages and bacon!',
              'The brown fox is quick and the blue dog is lazy!',
              'The sky is very blue and the sky is very beautiful today',
              'The dog is lazy but the brown fox is quick!'
    ]

     corpus = gutenberg.raw(fileids="carroll-alice.txt")
     corpus = sent_tokenize(corpus)

     corpus_preprocessor = TFIDFPreprocessor(simple_text_processor=True)
     corpus_preprocessor.load(path="data.pickle")
     results = corpus_preprocessor.process(corpus=corpus)

     for result in results:
         print(result)
     print(results.shape)



def main():
    process_example()








if __name__ == "__main__":
    main()


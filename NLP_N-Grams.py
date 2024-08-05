import nltk
import string
import re
from nltk.util import ngrams
from nltk import word_tokenize
from collections import Counter, defaultdict

text = "A proper society will constantly strive for equality. There are those who clamor for men and women to always be considered equal."
corpus = text.lower()
corpus = re.sub(r'\d+', '', corpus)
translator = str.maketrans('', '', string.punctuation)
corpus = corpus.translate(translator)
tokens = word_tokenize(corpus.lower())

unigrams = tokens
print("Unigrams model:")
print(unigrams)

bigrams = list(ngrams(tokens, 2))
print("\nBigrams model:")
print(bigrams)

trigrams = list(ngrams(tokens, 3))
print("\nTrigrams model:")
print(trigrams)

bigram_frequency = Counter(bigrams)
unigram_frequency = Counter(unigrams)
bigram_probability = {}
for (w1, w2), freq in bigram_frequency.items():
    bigram_probability[(w1, w2)] = freq / unigram_frequency[w1]
print("\nBigram Probabilities Model:")

for bigram, prob in bigram_probability.items():
    print(f"{bigram} : {prob}")

def predict_next_word(current_word, bigram_probability):
    candidates = {pair[1]: prob for pair, prob in bigram_probability.items() if pair[0] == current_word}
    if not candidates:
        return None
    next_word = max(candidates, key=candidates.get)
    return next_word

current_word = 'for'
predicted_word = predict_next_word(current_word, bigram_probability)
print(f"\nPredicted next word after '{current_word}': {predicted_word}")

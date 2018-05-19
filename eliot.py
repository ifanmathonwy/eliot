"""Eliot is a tool for poetry generation.

It works by generating words from a language model, constrained by another
model of the scansion and rhyme proper to a particular poetic form.

Currently it generates Shakespearian sonnets only. The language model is a
bigram model from NLTK and the poetry model uses regular expressions to
match appropriate stress sequences.

TODO: Add tests.
TODO: Handle Petrarchan sonnets.
TODO: Handle some kind of neural language model.
TODO: Allow N-grams of arbitrary order.
TODO: Predict the candidates based on the whole current word sequence.
TODO: Don't choose randomly from candidate set. Shuffled sample based on the distribution.
TODO: Implement some kind of PoemBuilder class.
"""

import random

from nltk import bigrams, ConditionalFreqDist
from nltk.corpus import gutenberg

import poems


class BigramWordCandidateProvider(object):
    """Provides candidate next words given a word using a bigram model."""

    def __init__(self, corpus):
        """Initializer of the BigramWordCandidateProvider.

        Args:
            corpus: An iterable of word strings.
        """
        _bigrams = bigrams(corpus)
        self._cfd = ConditionalFreqDist(_bigrams)

    def candidates(self, word_sequence):
        """Returns a list of candidate next words given a word sequence.
        """
        word = word_sequence[-1]
        candidates = [
            candidate for (candidate, _) in self._cfd[word].most_common()]
        return candidates

    def random_word(self):
        return random.choice(list(self._cfd.items()))[0]

def main():
    corpus = gutenberg.words('chesterton-brown.txt')
    provider = BigramWordCandidateProvider(corpus)
    sonnet = poems.Sonnet(provider)
    print(sonnet.generate())


if __name__ == "__main__":
    main()

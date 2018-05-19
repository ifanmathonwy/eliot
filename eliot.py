"""Eliot is a tool for poetry generation.

It works by generating words from a language model, constrained by another
model of the scansion and rhyme proper to a particular poetic form.

Currently it generates Shakespearian sonnets only. The language model is a
bigram model from NLTK and the poetry model uses regular expressions to
match appropriate stress sequences.

TODO: Add line generation method that works backwards from rhyming word.
TODO: After doing the above, implement a more efficient sonnet generation algorithm.
TODO: Handle Petrarchan sonnets.
TODO: Handle some kind of neural language model.
TODO: Cluster lines by meaning as well as rhyme.
TODO: Limericks.
TODO: Predict the candidates based on the whole current word sequence.
TODO: Don't choose randomly from candidate set. Shuffled sample based on the distribution.
"""

import itertools
import random
import re
import time
from collections import defaultdict

import pronouncing
from nltk import bigrams, ConditionalFreqDist
from nltk.corpus import gutenberg

PARTIAL_IAMBIC_RE = re.compile(r"^0$|^(0[12]){1,5}0?$|^(0[12]){1,4}0$")
FULL_IAMBIC_RE = re.compile(r"^(0[12]){5}0?$")

def stresses_for_word_sequence(word_sequence):
    """Gets the CMUDict stress sequence for a given word sequence.

    Args:
        word_sequence (list): A list of words.

    Returns:
        string: A stress sequence where 0 is zero stress, 1 is primary stress,
            and 2 is secondary stress.

    """
    stress_sequence = []
    for word in word_sequence:
        result = pronouncing.phones_for_word(word)
        if result:
            stress_sequence.append(pronouncing.stresses(result[0]))
        else:
            return ""
    return "".join(stress_sequence)

def is_partial_iambic(word_sequence):
    """Identifies a valid part of a line of iambic pentameter.

    An iamb is a metrical foot consisting of an unstressed (0 stress) syllable,
    followed by a stressed syllable (1 or 2 stress). Iambic pentameter is a
    sequence of five iambs, with the last iamb optionally having an extra
    unstressed syllable at the end.

    Args:
        word_sequence (list): A list of words.

    Returns:
        bool: True if word sequence is a valid partial iambic sequence. False
            otherwise.

    """
    stresses = stresses_for_word_sequence(word_sequence)
    if PARTIAL_IAMBIC_RE.match(stresses):
        return True
    else:

        return False

def is_full_iambic(word_sequence):
    """Identifies a full line of iambic pentameter.

    Args:
        word_sequence (list): A list of words.

    Returns:
        bool: True if word sequence is a valid full iambic sequence. False
            otherwise.

    """
    stresses = stresses_for_word_sequence(word_sequence)
    if FULL_IAMBIC_RE.match(stresses):
        return True
    else:
        return False


class BigramWordCandidateProvider(object):
    """Provides candidate next words given a word using a bigram model."""

    def __init__(self, corpus):
        """Initializer of the BigramWordCandidateProvider.

        Args:
            corpus: An iterable of word strings.
        """
        _bigrams = bigrams(corpus)
        self._cfd = ConditionalFreqDist(_bigrams)

    def candidates(self, word):
        """Returns a list of candidate next words given a word.
        """
        candidates = [
            candidate for (candidate, _) in self._cfd[word].most_common()]
        return candidates

    def random_word(self):
        return random.choice(list(self._cfd.items()))[0]

class GenerationTimeout(Exception):
    """Raised when generating a line of poetry takes too long."""
    pass

def generate_iambic_sentence(candidate_provider,
                             start_word=None,
                             timeout_seconds=10):
    """Generates a random iambic sentence.

    Args:
        candidate_provider: An object which implements candidates(word) and
            random_word() methods.
        start_word (string): The first word of the sentence. Not guaranteed to
            be used if no iambic sentence can be generated from it.
        timeout_seconds (int): number of seconds after which to give up on
            generating a random iambic sentence.

    Returns:
        list: The words of the iambic sentence.

    Raises:
        GenerationTimeout: When we fail in generating an iambic sentence before
            the supplied timeout.

    """
    dead_ends = defaultdict(list)
    word_sequence = []
    if not start_word:
        start_word = candidate_provider.random_word()
    word_sequence.append(start_word)
    timeout = time.time() + timeout_seconds
    while not is_full_iambic(word_sequence):
        candidates = candidate_provider.candidates(word_sequence[-1])
        random.shuffle(candidates)
        for candidate in candidates:
            if candidate in dead_ends["".join(word_sequence)]:
                continue
            extension = word_sequence + [candidate.lower()]
            if is_partial_iambic(extension):
                word_sequence.append(candidate)
                break
        else:
            if len(word_sequence) > 1:
                dead_ends["".join(word_sequence[:-1])].append(
                    word_sequence[-1])
                word_sequence.pop()
            else:
                word_sequence[0] = candidate_provider.random_word()
        if time.time() > timeout:
            raise GenerationTimeout(
                "Iambic sentence generation timed out after {} seconds.".
                    format(timeout_seconds))
    return word_sequence

class InsufficientSentencesError(Exception):
    """When the candidate pool isn't rich enough to generate a sonnet."""
    pass

def generate_sonnet(candidate_provider, candidate_pool_size=500):
    """Generates a Shakespearian sonnet of 14 lines of iambic pentameter.

    Args:
        candidate_provider: An object which implements candidates(word) and
            random_word() methods.

        candidate_pool_size: (int) The number of base candidate sentences out
            of which to construct the poem.

    Returns:
        string: The generated sonnet.
    """
    sentences = set()
    for _ in range(candidate_pool_size):
        sentence = generate_iambic_sentence(candidate_provider)
        sentences.add(" ".join(sentence))

    clusters = defaultdict(list)
    while len(list(filter(lambda c: len(c) > 2, clusters.values()))) < 7:
        try:
            sentence = sentences.pop()
        except KeyError:
            raise InsufficientSentencesError(
                "Candidate pool is not rich enough!")
        last_word = sentence.split(" ")[-1]
        last_word_phones = pronouncing.phones_for_word(last_word)[0]
        rhyming_part = pronouncing.rhyming_part(last_word_phones)
        key = rhyming_part
        if last_word not in [s.split(" ")[-1] for s in clusters[key]]:
            clusters[key].append(sentence)
    couplets = list(filter(lambda c: len(c) > 2, clusters.values()))
    random.shuffle(couplets)
    couplets = [random.sample(couplet, 2) for couplet in couplets]

    poem = []
    for a, b in zip((0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 6),
                    (0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1)):
        poem.append(couplets[a][b])
    return "\n".join(poem)

def main():
    corpus = itertools.chain(gutenberg.words('blake-poems.txt'),
                             gutenberg.words('austen-sense.txt'),
                             gutenberg.words('whitman-leaves.txt'))
    provider = BigramWordCandidateProvider(corpus)
    print(generate_sonnet(provider))


if __name__ == "__main__":
    main()
"""Eliot is a tool for poetry generation.

It works by generating words from a language model, constrained by another
model of the scansion and rhyme proper to a particular poetic form.

Currently it generates Shakespearian sonnets and Limericks only. The language model is a
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

from collections import defaultdict
import itertools
import logging
import random
import re
import time

from nltk import bigrams, ConditionalFreqDist
from nltk.corpus import gutenberg
import pronouncing

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


def stresses_for_word_sequence(word_sequence):
    """Gets the CMUdict stress sequence for a given word sequence.

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
            return ''
    return ''.join(stress_sequence)


def get_meter_validator(meter_re):
    """Returns a validator function for a regex matching a stress pattern."""
    def validator(word_sequence):
        stresses = stresses_for_word_sequence(word_sequence)
        if meter_re.match(stresses):
            return True
        else:
            return False

    return validator

class Validator(object):
    def __init__(self, partial_validator, full_validator):
        self.partial = partial_validator
        self.full = full_validator

# Iambic pentameter is used in the sonnet.
# An iamb is a metrical foot consisting of an unstressed (0 stress) syllable,
# followed by a stressed syllable (1 or 2 stress). Iambic pentameter is a
# sequence of five iambs, with the last iamb optionally having an extra
# unstressed syllable at the end.
PARTIAL_IAMBIC_PENTAMETER_RE = re.compile(r'^0$|^(0[12]){1,5}0?$')
FULL_IAMBIC_PENTAMETER_RE = re.compile(r'^(0[12]){5}0?$')

is_partial_iambic_pentameter = get_meter_validator(PARTIAL_IAMBIC_PENTAMETER_RE)
is_full_iambic_pentameter = get_meter_validator(FULL_IAMBIC_PENTAMETER_RE)
iambic_pentameter_validator = Validator(is_partial_iambic_pentameter,
                                        is_full_iambic_pentameter)

# Anapaestic dimeter and trimeter are used in the limerick.
PARTIAL_ANAPAESTIC_DIMETER_RE = re.compile(r'^00?$|^(00?[12])((00[12])0?)?$')
FULL_ANAPAESTIC_DIMETER_RE = re.compile(r'^(00?[12])(00[12])0?$')
PARTIAL_ANAPAESTIC_TRIMETER_RE = re.compile(r'^00?$|^(00?[12])(00[12]){1,2}0?$')
FULL_ANAPAESTIC_TRIMETER_RE = re.compile(r'^(00?[12])(00[12]){2}0?$')

is_partial_anapaestic_dimeter = get_meter_validator(PARTIAL_ANAPAESTIC_DIMETER_RE)
is_full_anapaestic_dimeter = get_meter_validator(FULL_ANAPAESTIC_DIMETER_RE)
anapaestic_dimeter_validator = Validator(is_partial_anapaestic_dimeter,
                                         is_full_anapaestic_dimeter)
is_partial_anapaestic_trimeter = get_meter_validator(PARTIAL_ANAPAESTIC_TRIMETER_RE)
is_full_anapaestic_trimeter = get_meter_validator(FULL_ANAPAESTIC_TRIMETER_RE)
anapaestic_trimeter_validator = Validator(is_partial_anapaestic_trimeter,
                                          is_full_anapaestic_trimeter)

class GenerationTimeout(Exception):
    """Raised when generating a line of poetry takes too long."""
    pass


def generate_metered_sentence(candidate_provider,
                              partial_validator,
                              full_validator,
                              start_word=None,
                              timeout_seconds=10):
    """Generates a random metered sentence.

    Args:
        candidate_provider: An object which implements candidates(sequence) and
            random_word() methods.
        partial_validator: A function which returns True when its input is a
            valid partial metered sentence.
        full_validator: A function which returns True when its input is a full
            metered sentence.
        start_word (string): The first word of the sentence. Not guaranteed to
            be used if no metered sentence can be generated from it.
        timeout_seconds (int): number of seconds after which to give up on
            generating a random metered sentence.

    Returns:
        list: The words of the metered sentence.

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
    while not full_validator(word_sequence):
        candidates = candidate_provider.candidates(word_sequence)
        random.shuffle(candidates)
        for candidate in candidates:
            if candidate in dead_ends[''.join(word_sequence)]:
                continue
            extension = word_sequence + [candidate.lower()]
            if partial_validator(extension):
                word_sequence.append(candidate)
                break
        else:
            if len(word_sequence) > 1:
                dead_ends[''.join(word_sequence[:-1])].append(
                    word_sequence[-1])
                word_sequence.pop()
            else:
                word_sequence[0] = candidate_provider.random_word()
        if time.time() > timeout:
            raise GenerationTimeout(
                'Metered sentence generation timed out after {} seconds.'.format(timeout_seconds))
    return word_sequence

def get_rhyming_groups(group_size, number_groups, pool):
    """Returns a list of rhyming groups of the given size from the given candidate pool.

    Args:
        group_size (int): number of lines in the rhyming group.
        number_groups (int): number of rhyming groups.
        pool (list) : candidate pool from which to draw lines.

    Raises:
         InsufficientSentencesError: if the candidate pool is not rich enough.
    """
    clusters = defaultdict(list)
    while len(list(filter(lambda c: len(c) > group_size, clusters.values()))) < number_groups:
        try:
            sentence = pool.pop()
        except KeyError:
            raise InsufficientSentencesError(
                'Candidate pool is not rich enough!')
        last_word = sentence.split(" ")[-1]
        last_word_phones = pronouncing.phones_for_word(last_word)[0]
        rhyming_part = pronouncing.rhyming_part(last_word_phones)
        key = rhyming_part
        if last_word not in [s.split(" ")[-1] for s in clusters[key]]:
            clusters[key].append(sentence)
    groups = list(filter(lambda c: len(c) > group_size, clusters.values()))
    random.shuffle(groups)
    return [random.sample(group, group_size) for group in groups]


def generate_candidate_pool(size, provider, validator):
    pool = set()
    for _ in range(size):
        sentence = generate_metered_sentence(provider,
                                             validator.partial,
                                             validator.full)
        print("generated ", ' '.join(sentence))
        pool.add(' '.join(sentence))
    return pool

class InsufficientSentencesError(Exception):
    """When the candidate pool isn't rich enough to generate a poem."""
    pass


class Sonnet(object):
    """Encapsulates logic for generating a Shakespearian sonnet."""

    def __init__(self, candidate_provider, candidate_pool_size=500):
        self.provider = candidate_provider
        self.candidate_pool_size = candidate_pool_size
        self.form_name = "Shakespearian sonnet."

    def generate(self):
        logging.info("Generating a {}...".format(self.form_name))
        sentences = generate_candidate_pool(self.candidate_pool_size,
                                            self.provider,
                                            iambic_pentameter_validator)
        group_dict = dict(zip(list('abcdefg'), get_rhyming_groups(2, 7, sentences)))
        poem = []
        scheme = ['a', 'b', 'a', 'b', 'c', 'd', 'c', 'd', 'e', 'f', 'e', 'f', 'g', 'g']
        for letter in scheme:
            poem.append(group_dict[letter].pop())
        return "\n".join(poem)


class Limerick(object):
    """Encapsulates logic for generating a limerick."""

    def __init__(self,
                 candidate_provider,
                 candidate_pool_a_size=300,
                 candidate_pool_b_size=200):
        self.provider = candidate_provider
        self.candidate_pool_a_size = candidate_pool_a_size
        self.candidate_pool_b_size = candidate_pool_b_size
        self.form_name = "limerick"

    def generate(self):
        logging.info("Generating a {}...".format(self.form_name))
        trimeter = generate_candidate_pool(self.candidate_pool_a_size,
                                           self.provider,
                                           anapaestic_trimeter_validator)
        dimeter = generate_candidate_pool(self.candidate_pool_b_size,
                                          self.provider,
                                          anapaestic_dimeter_validator)

        group_dict = {}
        # Rhyme scheme is AABBA. We need a triplet of A and a couplet of B.
        # First find the couplet of B.
        group_dict['b'] = get_rhyming_groups(2, 1, dimeter)[0]

        # Now find the triplet of A.
        group_dict['a'] = get_rhyming_groups(3, 1, trimeter)[0]

        poem = []
        scheme = ['a', 'a', 'b', 'b', 'a']
        for letter in scheme:
            poem.append(group_dict[letter].pop())
        return "\n".join(poem)

def main():
    corpus = itertools.chain(gutenberg.words('blake-poems.txt'),
                             gutenberg.words('austen-sense.txt'),
                             gutenberg.words('whitman-leaves.txt'))
    provider = BigramWordCandidateProvider(corpus)
    sonnet = Sonnet(provider)
    print(sonnet.generate())
    limerick = Limerick(provider)
    print(limerick.generate())


if __name__ == "__main__":
    main()

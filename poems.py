"""Classes for generating poems of various meters.
"""

import random
import re
import time
from collections import defaultdict

import pronouncing


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


# Iambic pentameter is used in the sonnet.
# An iamb is a metrical foot consisting of an unstressed (0 stress) syllable,
# followed by a stressed syllable (1 or 2 stress). Iambic pentameter is a
# sequence of five iambs, with the last iamb optionally having an extra
# unstressed syllable at the end.
PARTIAL_IAMBIC_PENTAMETER_RE = re.compile(r'^0$|^(0[12]){1,5}0?$')
FULL_IAMBIC_PENTAMETER_RE = re.compile(r'^(0[12]){5}0?$')

is_partial_iambic_pentameter = get_meter_validator(PARTIAL_IAMBIC_PENTAMETER_RE)
is_full_iambic_pentameter = get_meter_validator(FULL_IAMBIC_PENTAMETER_RE)

# Anapaestic dimeter and trimeter are used in the limerick.
PARTIAL_ANAPAESTIC_DIMETER_RE = re.compile(r'^00?$|^(00?[12])((00[12])0?)?$')
FULL_ANAPAESTIC_DIMETER_RE = re.compile(r'^(00?[12])(00[12])0?$')
PARTIAL_ANAPAESTIC_TRIMETER_RE = re.compile(r'^00?$|^(00?[12])(00[12]){1,2}0?$')
FULL_ANAPAESTIC_TRIMETER_RE = re.compile(r'^(00?[12])(00[12]){2}0?$')

is_partial_anapaestic_dimeter = get_meter_validator(PARTIAL_ANAPAESTIC_DIMETER_RE)
is_full_anapaestic_dimeter = get_meter_validator(FULL_ANAPAESTIC_DIMETER_RE)
is_partial_anapaestic_trimeter = get_meter_validator(PARTIAL_ANAPAESTIC_TRIMETER_RE)
is_full_anapaestic_trimeter = get_meter_validator(FULL_ANAPAESTIC_TRIMETER_RE)


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
        print(' '.join(word_sequence))
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


class InsufficientSentencesError(Exception):
    """When the candidate pool isn't rich enough to generate a poem."""
    pass


class Sonnet(object):
    """TODO: Add documentation."""
    def __init__(self, candidate_provider, candidate_pool_size=500):
        self.provider = candidate_provider
        self.candidate_pool_size = candidate_pool_size

    def generate(self):
        sentences = set()
        for _ in range(self.candidate_pool_size):
            sentence = generate_metered_sentence(self.provider,
                                                 is_partial_iambic_pentameter,
                                                 is_full_iambic_pentameter)
            sentences.add(" ".join(sentence))

        clusters = defaultdict(list)
        while len(list(filter(lambda c: len(c) > 2, clusters.values()))) < 7:
            try:
                sentence = sentences.pop()
            except KeyError:
                raise InsufficientSentencesError(
                    'Candidate pool is not rich enough!')
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


class Limerick(object):
    """TODO: Add documentation."""
    def __init__(self,
                 candidate_provider,
                 candidate_pool_a_size=300,
                 candidate_pool_b_size=200):
        self.provider = candidate_provider
        self.candidate_pool_a_size = candidate_pool_a_size
        self.candidate_pool_b_size = candidate_pool_b_size

    def generate(self):
        trimeter = set()
        dimeter = set()
        for _ in range(self.candidate_pool_a_size):
            sentence = generate_metered_sentence(self.provider,
                                                 is_partial_anapaestic_trimeter,
                                                 is_full_anapaestic_trimeter)
            trimeter.add(' '.join(sentence))
        for _ in range(self.candidate_pool_b_size):
            sentence = generate_metered_sentence(self.provider,
                                                 is_partial_anapaestic_dimeter,
                                                 is_full_anapaestic_dimeter)
            dimeter.add(' '.join(sentence))

        # Rhyme scheme is AABBA. We need a triplet of A and a couplet of B.
        # First find the couplet of B.
        dimeter_clusters = defaultdict(list)
        while not list(filter(lambda c: len(c) > 2, dimeter_clusters.values())):
            try:
                sentence = dimeter.pop()
            except KeyError:
                raise InsufficientSentencesError(
                    "Candidate pool is not rich enough!")
            last_word = sentence.split(" ")[-1]
            last_word_phones = pronouncing.phones_for_word(last_word)[0]
            rhyming_part = pronouncing.rhyming_part(last_word_phones)
            key = rhyming_part
            if last_word not in [s.split(" ")[-1] for s in dimeter_clusters[key]]:
                dimeter_clusters[key].append(sentence)
        couplet = list(filter(lambda c: len(c) > 2, dimeter_clusters.values()))[0]

        # Now find the triplet of A
        trimeter_clusters = defaultdict(list)
        while not list(filter(lambda c: len(c) > 3, trimeter_clusters.values())):
            try:
                sentence = trimeter.pop()
            except KeyError:
                raise InsufficientSentencesError(
                    "Candidate pool is not rich enough!")
            last_word = sentence.split(" ")[-1]
            last_word_phones = pronouncing.phones_for_word(last_word)[0]
            rhyming_part = pronouncing.rhyming_part(last_word_phones)
            key = rhyming_part
            if last_word not in [s.split(" ")[-1] for s in trimeter_clusters[key]]:
                trimeter_clusters[key].append(sentence)
        triplet = list(filter(lambda c: len(c) > 3, trimeter_clusters.values()))[0]
        poem = [triplet[0],
                triplet[1],
                couplet[0],
                couplet[1],
                triplet[2]]
        return "\n".join(poem)
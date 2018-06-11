"""Eliot is a library for poetry generation.

It works by generating words from a language model, constrained by another
model of the scansion and rhyme proper to a particular poetic form.

The language model is a bigram model from NLTK and the poetry model uses
regular expressions to match appropriate stress sequences.

TODO: tests.
TODO: optional local caching/pickling of candidate pool.
TODO: verbose mode logging generation process.
"""

from collections import defaultdict
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


class Meter(object):
    """Object with full and partial methods for validating a line of meter."""

    def __init__(self, partial_validator, full_validator):
        """Initializer for the Meter object.

        Args:
            partial_validator: A method which returns True iff its input is a
                valid partial line of the given meter.
            full_validator: A method which returns True iff its input is a
                valid full line of the given meter.
        """
        self.partial = partial_validator
        self.full = full_validator


# Iambic pentameter is used in the sonnet.
# An iamb is a metrical foot consisting of an unstressed (0 stress) syllable,
# followed by a stressed syllable (1 or 2 stress). Iambic pentameter is a
# sequence of five iambs, with the last iamb optionally having an extra
# unstressed syllable at the end.
PARTIAL_IAMBIC_PENTAMETER_RE = re.compile(r'^0$|^(0[12]){1,5}0?$')
FULL_IAMBIC_PENTAMETER_RE = re.compile(r'^(0[12]){5}0?$')

_is_partial_iambic_pentameter = get_meter_validator(PARTIAL_IAMBIC_PENTAMETER_RE)
_is_full_iambic_pentameter = get_meter_validator(FULL_IAMBIC_PENTAMETER_RE)
IAMBIC_PENTAMETER = Meter(_is_partial_iambic_pentameter, _is_full_iambic_pentameter)

# Anapaestic dimeter and trimeter are used in the limerick.
PARTIAL_ANAPAESTIC_DIMETER_RE = re.compile(r'^00?$|^(00?[12])((00[12])0?)?$')
FULL_ANAPAESTIC_DIMETER_RE = re.compile(r'^(00?[12])(00[12])0?$')
PARTIAL_ANAPAESTIC_TRIMETER_RE = re.compile(r'^00?$|^(00?[12])(00[12]){1,2}0?$')
FULL_ANAPAESTIC_TRIMETER_RE = re.compile(r'^(00?[12])(00[12]){2}0?$')

_is_partial_anapaestic_dimeter = get_meter_validator(PARTIAL_ANAPAESTIC_DIMETER_RE)
_is_full_anapaestic_dimeter = get_meter_validator(FULL_ANAPAESTIC_DIMETER_RE)
ANAPAESTIC_DIMETER = Meter(_is_partial_anapaestic_dimeter, _is_full_anapaestic_dimeter)

_is_partial_anapaestic_trimeter = get_meter_validator(PARTIAL_ANAPAESTIC_TRIMETER_RE)
_is_full_anapaestic_trimeter = get_meter_validator(FULL_ANAPAESTIC_TRIMETER_RE)
ANAPAESTIC_TRIMETER = Meter(_is_partial_anapaestic_trimeter, _is_full_anapaestic_trimeter)


class GenerationTimeout(Exception):
    """Raised when generating a line of poetry takes too long."""
    pass


def generate_metered_sentence(candidate_provider,
                              partial_validator,
                              full_validator,
                              start_word=None,
                              timeout_seconds=20):
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


def generate_candidate_pool(size, provider, meter):
    """Generates a pool of candidate lines from the given provider.

    Args:
        size (int): Number of candidate lines in the pool.
        provider: An object which implements candidates(sequence) and
            random_word() methods.
        meter: An object which implements full and partial meter
            validator methods.

    Returns:
        pool (set): A set of candidate lines fitting the given meter.
    """
    pool = set()
    for _ in range(size):
        sentence = generate_metered_sentence(provider,
                                             meter.partial,
                                             meter.full)
        pool.add(' '.join(sentence))
    return pool


class InsufficientSentencesError(Exception):
    """When the candidate pool isn't rich enough to generate a poem."""
    pass


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
    while len(list(filter(lambda c: len(c) >= group_size, clusters.values()))) < number_groups:
        try:
            sentence = pool.pop()
        except KeyError:
            raise InsufficientSentencesError(
                'Candidate pool is not rich enough!')
        last_word = sentence.split(" ")[-1]
        last_word_phones = pronouncing.phones_for_word(last_word)[0]
        rhyming_part = pronouncing.rhyming_part(last_word_phones)
        if last_word not in [s.split(" ")[-1] for s in clusters[rhyming_part]]:
            clusters[rhyming_part].append(sentence)
    groups = list(filter(lambda c: len(c) >= group_size, clusters.values()))
    random.shuffle(groups)
    return [random.sample(group, group_size) for group in groups]


class PoemDesignError(Exception):
    """Raised when a poem design is ill-formed."""
    pass


class Poem(object):
    """Encapsulates the logic for generating a poem."""

    def __init__(self, candidate_provider):
        """Initializer of the Poem object.

        Args:
            candidate_provider: An object which implements
                candidates(sequence) and random_word() methods.
        """
        # TODO: Add way to specify line type and schemes at initialization.
        self._provider = candidate_provider
        self._meters = {}
        self._pool_sizes = {}
        self._scheme_to_type = defaultdict(lambda: None)
        self._scheme_counter = defaultdict(int)
        self._candidate_pools = {}
        self._type_to_scheme = defaultdict(set)

    def register_line_type(self, name, meter, candidate_pool_size=600):
        """Register a type of line that can be used in designing the poem.

        Args:
            name (string): The name of the line type.
            meter: An object which implements full and partial validation
                methods.
            candidate_pool_size (int): The size of the candidate pool to be
                used when generating lines of this type.

        """
        self._meters[name] = meter
        self._pool_sizes[name] = candidate_pool_size

    def design(self, rhyme_scheme, type_scheme):
        """Specify the meter of each line as well as the rhyme scheme.

        Args:
            rhyme_scheme (list): A list of the rhyme groups for each line
                of the poem. If two lines have the same rhyme group, they
                rhyme.
            type_scheme (list): A list of the line types for each line of
                the poem. These line types should be registered.

        Raises:
            PoemDesignError: When two lines are in the same rhyme group
                but do not have the same line type.
        """
        self._rhyme_scheme = rhyme_scheme
        self._type_scheme = type_scheme
        assert len(self._rhyme_scheme) == len(self._type_scheme)
        for rhyme, type in zip(self._rhyme_scheme, self._type_scheme):
            if self._scheme_to_type[rhyme] and self._scheme_to_type[rhyme] != type:
                raise PoemDesignError('Poem design is ill-formed.')
            else:
                self._scheme_to_type[rhyme] = type
                self._scheme_counter[rhyme] += 1
        # For each scheme, find out its type and its count.
        for rhyme, count in self._scheme_counter.items():
            type = self._scheme_to_type[rhyme]
            self._type_to_scheme[(type, count)].add(rhyme)

    def generate(self):
        """Generates the poem.

        Returns:
             A a list of the lines of the poem.
        """
        if not self._candidate_pools:
            for type in self._meters:
                self._candidate_pools[type] = generate_candidate_pool(self._pool_sizes[type],
                                                                      self._provider,
                                                                      self._meters[type])
        group_dict = {}
        # Work backwards from what is needed at the end of the function to deobfuscate
        # the function.
        for (type, count), rhymes in self._type_to_scheme.items():
            rhyming_groups = get_rhyming_groups(count,
                                                len(rhymes),
                                                self._candidate_pools[type])
            for rhyme in rhymes:
                group_dict[rhyme] = rhyming_groups.pop()

        poem = []
        for letter in self._rhyme_scheme:
            poem.append(group_dict[letter].pop())
        return poem


def generate_sonnet(provider):
    """Print a sonnet from the given provider."""
    sonnet = Poem(provider)
    sonnet.register_line_type('iambic pentameter', IAMBIC_PENTAMETER)
    sonnet.design(['a', 'b', 'a', 'b', 'c', 'd', 'c', 'd', 'e', 'f', 'e', 'f', 'g', 'g'],
                  ['iambic pentameter'] * 14)
    generated_sonnet = sonnet.generate()
    print('\n'.join(generated_sonnet))


def generate_limerick(provider):
    """Print a limerick from the given provider."""
    limerick = Poem(provider)
    limerick.register_line_type('anapaestic dimeter',
                                ANAPAESTIC_DIMETER,
                                candidate_pool_size=200)
    limerick.register_line_type('anapaestic trimeter',
                                ANAPAESTIC_TRIMETER,
                                candidate_pool_size=250)
    limerick.design(['a', 'a', 'b', 'b', 'a'],
                    ['anapaestic trimeter',
                     'anapaestic trimeter',
                     'anapaestic dimeter',
                     'anapaestic dimeter',
                     'anapaestic trimeter'])
    generated_limerick = limerick.generate()
    print('\n'.join(generated_limerick))


def main():
    corpus = gutenberg.words('whitman-leaves.txt')
    provider = BigramWordCandidateProvider(corpus)
    generate_sonnet(provider)
    generate_limerick(provider)


if __name__ == '__main__':
    main()

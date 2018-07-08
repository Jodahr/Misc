#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing some useful preprocessing functions
for natural language processing."""

import re
import string
from operator import itemgetter
import numpy as np
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# # nltk downloads
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')


def rem_html(text):
    """Removes html from text.

    Args:
        text (string): Raw text of type string including html statements.

    Returns:
        String without any html statements.

    """

    return BeautifulSoup(text, 'html.parser').get_text()


def rem_stop(token_list, language='english', wordlist=[]):
    """Removes stopwords of a given language and a custom stopwords
    from a list of strings.

    Args:
        token_list (list): List of strings, i.e. tokenized text.
        language (string): Language used for nltk stopwords.
        wordlist (list): List of custom stopwords.

    Returns:
        List of strings with removed stopword elements.

    """

    return [word for word in token_list
            if word.lower() not in
            stopwords.words(language) + wordlist]


def stem_words(token_list, language='english'):
    """
    Uses nltk snowball stemming to stem list of words.

    Args:
        token_list (list): List of strings which need to be stemmed.
        language (string): Language which will be used for stemming.

    Returns:
        List of stemmed strings.

    """
    stemmer = SnowballStemmer(language)
    return [stemmer.stem(element) for element in token_list]


def rem_punctuation(text):
    """Removes chars contained in string.punctuation from each string.

    Args:
        text (string): String which will be cleaned from punctuation.

    Returns:
        String without any punctuation chars.

    """

    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub(' ', text)


def keep_string_printable(text):
    """Removes chars not contained in string.printable from each string.

    Args:
        text (string): String which will be cleaned from punctuation.

    Returns:
        String without any unprintable chars.

    """

    printable = set(string.printable)
    # filter returns iterable
    return ''.join(filter(lambda x: x in printable, text))


def replace_umlaute(text):
    text = text.replace(u'ü', 'ue')
    text = text.replace(u'ö', 'oe')
    text = text.replace(u'ä', 'ae')
    text = text.replace(u'ß', 'ss')
    return text


def rem_additional_whitespaces(text):
    """Removes additional and trailing whitespaces from text.

    Args:
        text (string): Text string which will get cleaned from
            additional whitespaces.

    Returns:
        String without additional and trailing whitespaces.

    """

    return re.sub('\s+', ' ', text).strip()


def get_wordnet_pos(treebank_tag):
    """Translates treebank_tag to wordnet position in order to use lemmatizer.

    Args:
        treebank_tag (string): Penn treebank tag generated with nltk.pos_tag.

    Returns:
        Translated tag string to wordnet position.

    """

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_words(token_list, language='english'):
    """Lemmatizing list of words. Only working for english words so far.

    Args:
        token_list (list): List of words which will get lemmatized.

    Returns:
        Lemmatized list of words.

    """

    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer(language)
    penn_treebank_tag_list = nltk.pos_tag(token_list)
    lemmatized = []
    for tuples in penn_treebank_tag_list:
        pos = get_wordnet_pos(tuples[1])
        if pos:
            lemmatized.append(lemmatizer.lemmatize(tuples[0], pos))
        else:
            lemmatized.append(stemmer.stem(tuples[0]))
    return lemmatized


# Can be implemented in a more efficient way, e.g. less loops over the doc
# by combining cleaning steps in one loop.
def text_preprocess(text, stemming=True, html=True,
                    stop=True, language='english',
                    umlaute=True, keep_printable=True, concat=False):
    """ Combination of different preprocessing steps for text mining.
    The steps are:

    1.)  remove html (optional)
    2.)  transform string to lowercase
    3.)  remove punctuation
    4.)  replace umlaute (optional)
    5.)  keep printable characters (optional)
    6.)  remove additional whitespaces
    7.)  tokenize sentence into list of tokens
    8.)  remove stopwords (optional)
    9.)  stem or lemmatize words. default: stemming
    10.) concat tokens (optional)

    Args:
        text (string): A sentence or a document.

    Returns:
        Cleaned tokenized list of string
        or single string if concat is set to True.

    """

    if html:
        text = rem_html(text)
    text = text.lower()
    text = rem_punctuation(text)
    if umlaute:
        text = replace_umlaute(text)
    if keep_printable:
        text = keep_string_printable(text)
    text = rem_additional_whitespaces(text)
    text_list = nltk.word_tokenize(text)
    if stemming or language != 'english':
        text_list = stem_words(text_list, language)
    else:
        text_list = lemmatize_words(text_list)
    if stop:
        text_list = rem_stop(text_list, language)
    if concat:
        return ' '.join(text_list)
    return text_list


def flatten_list(nested_list):
    """ Flattens a list which includes one nested list:

    Args:
        nested_list (list): List of lists.
    Returns:
        Flatten list.

    """

    return [item for sublist in nested_list for item in sublist]


def get_vocabulary(corpus, flatten=True):
    """ Computes a word_to_index and index_to_word vocabulary.

    Args:
        corpus (list): List of documents.
    Returns:
        Vocabulary dictionary and inverse dictionary.
        The index corresponds to the order of wordcount frequency.

    """

    if flatten:
        flat_list = flatten_list(corpus)
    else:
        flat_list = corpus
    token_set = set(flat_list)
    vocab_size = len(token_set)
    words, counts = wordcount_corpus(flat_list, flatten=False)
    sorted_tuples = np.array(sorted(zip(words, counts),
                                    key=itemgetter(1), reverse=True))
    indices = np.arange(vocab_size) + 3
    word_to_index = dict(zip(sorted_tuples[:, 0], indices))
    word_to_index['<PAD>'] = 0
    word_to_index['<START>'] = 1
    word_to_index['<UNKNOWN>'] = 2
    index_to_word = {value: key for key, value in word_to_index.items()}
    return word_to_index, index_to_word


def wordcount_corpus(corpus, flatten=True):
    """ Computes a wordcount over the corpus.

    Args:
        corpus (list): List of documents.
    Returns:
        Two numpy arrays containing the words and the corresponding wordcounts.

    """

    if flatten:
        flat_list = flatten_list(corpus)
    else:
        flat_list = corpus
    flat_array = np.array(flat_list)
    return np.unique(flat_array, return_counts=True)


class CorpusEncoding():

    def __init__(self):
        self.word_to_index = None
        self.index_to_word = None
        self.vocab_size = None

    def fit(self, tokenized_corpus):
        self.word_to_index, self.vocab_size = get_vocabulary(tokenized_corpus,
                                                             flatten=True)
        self.vocab_size = len(self.word_to_index)

    def reduce_vocab(self, lower_rank, upper_rank=None):
        if not upper_rank:
            upper_rank = 0
        self.word_to_index = {k: v for k, v in self.word_to_index.items()
                              if v <= lower_rank and v >= upper_rank}
        self.index_to_word = {value: key for key, value
                              in self.word_to_index.items()}

    def transform(self, tokenized_corpus, drop_unknown=False):
        encoded_tokenized_corpus = []
        for doc in tokenized_corpus:
            encoded_doc = [self.word_to_index[word]
                           if word in self.word_to_index
                           else 2 for word in doc]
            encoded_doc.insert(0, 1)
            if drop_unknown:
                encoded_doc = [element for element in encoded_doc
                               if element != 2]
            encoded_tokenized_corpus.append(encoded_doc)
        return encoded_tokenized_corpus

    def inverse_transform(self, encoded_tokenized_corpus):
        tokenized_corpus = []
        for encoded_doc in encoded_tokenized_corpus:
            decoded_doc = [self.index_to_word[index] for index in encoded_doc]
            tokenized_corpus.append(decoded_doc)
        return tokenized_corpus


if __name__ == '__main__':
    print("Module test:")
    sentence = "<br>The quick  brown brown fox jumps over the lazy dogs. </br>"
    print("The sentence which will be used for testing is:\n\n{} \n"
          .format(sentence))

    print("1. remove html:")
    string_prep = rem_html(sentence)
    print(string_prep)

    print("2. to lower:")
    string_prep = string_prep.lower()
    print(string_prep)

    print("3. rm punctuation:")
    string_prep = rem_punctuation(string_prep)
    print(string_prep)

    print("4. rm additional whitespaces:")
    string_prep = rem_additional_whitespaces(string_prep)
    print(string_prep)

    print("5. tokenize:")
    string_prep = nltk.word_tokenize(string_prep)
    print(string_prep)

    print("6. remove stopwords:")
    string_prep = rem_stop(string_prep)
    print(string_prep)

    print("7a. stemming:")
    string_prepA = stem_words(string_prep)
    print(string_prepA)

    print("7b. lemmatizing:")
    string_prepB = lemmatize_words(string_prep)
    print(string_prepB)

    print("full preprocessing function:")
    string_prep = text_preprocess(sentence)
    print(string_prep)

    # print("full preprocessing function:")
    # string_prep = text_preprocess(sentence, stemming=False, concat=True)
    # print(string_prep)

    print(get_vocabulary(string_prep, flatten=False))
    print(wordcount_corpus(string_prep, flatten=False))

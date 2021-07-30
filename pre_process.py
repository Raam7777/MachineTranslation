import unicodedata
import re
from pickle_funcs import create_pickle
from global_vars import _MAX_LENGTH
from global_vars import _lang1
from global_vars import _lang2
from global_vars import _reverse


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Zאבגדהוזחטיכלמנסעפצקרשתךםןףץ.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, data, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(data % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filter_pair(p):
    return len(p[0].split(' ')) < _MAX_LENGTH and \
           len(p[1].split(' ')) < _MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, data, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, data, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class PreProcessData:
    def __init__(self, lang1, lang2, data, reverse):
        self.input_lang, self.output_lang, self.pairs = prepare_data(lang1, lang2, data, reverse)

        name = self.input_lang.name + "-" + self.output_lang.name + "-dictionary.pickle"
        create_pickle(name, self)


def main():
    PreProcessData(_lang1, _lang2, 'data/%s-%s.txt', _reverse)


if __name__ == '__main__':
    main()
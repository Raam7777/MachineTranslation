from pickle_funcs import load_pickle
from algos import levenshtein_distance
from pre_process import PreProcessData
from pre_process import Lang


dictionary = load_pickle("dictionaries/eng-fra-dictionary.pickle")
test_sentence = "he gave it"

sents = []
for i, j in dictionary.pairs:
    sents.append(i)

result = levenshtein_distance(test_sentence,sents)
print(result)
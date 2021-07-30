import Levenshtein as lev


def levenshtein_distance(s, words):
    min_distance = 1000
    best_word = ""

    for i in range(len(words)):
        distance = lev.distance(s, words[i])
        if distance < min_distance:
            best_word = words[i]
            min_distance = distance
    return best_word

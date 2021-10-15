#!/usr/bin/env python3

import random
import re
import sys
import json
import math


def preprocess_line(line: str):
    """
    Takes a line of text and returns a new string without characters outside of the set: characters in the English
    alphabet, space, digits, or the ‘.’ character. Additionally the function lowercase all the letters and
    substitutes all digits with '0'.
    :param line:
    :return: line without irrelevant characters
    """
    return re.sub(r"\d", '0', re.sub(r"[^a-zA-Z\d. ]", '', line).lower()) + "\n"


def generate_n_gram_model(corpus: str, n: int):
    """
    Create a MLE model for ngram of size n on given corpus.
    :param corpus: String to train the model
    :param n: Size of ngram
    :return: Dictionary representing the model. The dictionary maps from a string key of size n-1 to another dictionary
                of characters observed after that string and their count, which is not normalized at this point
    """
    size = len(corpus)
    # Surround the line with special character indicating start and end of line
    corpus = '#' * (n - 1) + corpus + '#'

    if size == 0:
        return {}

    # Get the counts
    counts = dict()
    for i in range(n, size + n + 1):
        ngram = corpus[i - n: i]
        prev = ngram[0: n - 1]
        char = ngram[n - 1]
        if prev not in counts:
            counts[prev] = dict()
        if char not in counts[prev]:
            counts[prev][char] = 0
        counts[prev][char] += 1

    return counts


def train_model(file: str, n: int):
    fp = open(file, 'r')
    probabilities = dict()
    for line in fp:
        counts = generate_n_gram_model(preprocess_line(line), n)
        for condition in counts:
            if condition not in probabilities:
                probabilities[condition] = dict()
            for character in counts[condition]:
                if character not in probabilities[condition]:
                    probabilities[condition][character] = 0
                probabilities[condition][character] += counts[condition][character]

    # Normalize the counts in the dictionary to get actual probabilities
    for condition in probabilities:
        total_count = sum(probabilities[condition].values())
        for character in probabilities[condition]:
            probabilities[condition][character] /= total_count

    #save_model("{}_{}gram_model".format(file, n), probabilities)
    fp.close()
    return probabilities


def generate_from_ML(model: dict, n: int):
    condition = "##"
    result = ""
    count = 0
    while count < n:
        population = list(model[condition].keys())
        weights = list(model[condition].values())
        choice = random.choices(population=population, weights=weights, k=1)[0]
        # if the model chooses to start new sentence
        if choice == '#':
            condition = "##"  # reset the condition to start a new sentence
            continue
        count += 1
        result += choice
        condition = condition[1] + choice
    return result


def save_model(file: str, model: dict):
    fp = open(file, 'w')
    for condition in model:
        for character in model[condition]:
            line = "{}{}\t{}\n".format(condition, character, model[condition][character])
            fp.write(line)
    fp.close()

# Given a Language Model, a Test Documents path as a string, and the n-gram n, return the perplexity
# Perplexity of a sequence W (i.e. PP(W)) is given by equation
#PP(W) 2 ^ l
# where l = (-1/N) * log(P(W_n))  i.e. (-1/(length of the sequence)) * (the log of the propbability of the sequence)
def compute_perplexity(model: dict, doc: str, n: int):
    sum_of_logs = 0.0
    model
    corpus = open(doc, "r")
    cleaned_corpus = ""
    for line in corpus:
        cleaned_corpus += preprocess_line(line)

    cleaned_corpus_size = len(cleaned_corpus)
    for i in range(n, cleaned_corpus_size + 1):
        ngram = cleaned_corpus[i - n: i]
        given = ngram[0: n - 1]
        current = ngram[n - 1]
        try:
            sum_of_logs += math.log(model[given][current])
        except:
            continue

    entropy = (-1.0/cleaned_corpus_size) * sum_of_logs
    perplexity = 2**(entropy)
    print("perplexity of given model on " + doc+" : "+ str(perplexity))



if __name__ == '__main__':
    training_filename = sys.argv[1]
    perplex_filename = sys.argv[2]
    # n_gram = int(sys.argv[2])
    # n_generate = int(sys.argv[3])

    n_gram = 3
    n_generate = 300

    language_model = train_model(training_filename, n_gram)
    compute_perplexity(language_model, perplex_filename, n_gram)
    #print(generate_from_ML(language_model, n_generate))

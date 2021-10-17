#!/usr/bin/env python3

import random
import re
import sysimport json
import math
import matplotlib.pylab as plt


# Remove any symbols than are not english characters, space, coma or a digit
# Additionally change all digits to '0' and make all characters lowercase
def preprocess_line(line: str):
    return re.sub(r"\d", '0', re.sub(r"[^a-zA-Z\d. ]", '', line).lower())


def generate_all_trigrams(vocab: str):
    # Generate all trigrams possible with given vocabulary
    result = set()
    # Add trigrams of form ##c1, which begin the sentence
    for c1 in vocab:
        result.add("##" + c1)
    # Add trigrams of form #c1c2, first two characters of a sentence
    for c1 in vocab:
        for c2 in vocab:
            result.add("#" + c1 + c2)
    # Add trigrams of form c1c2c3, any combination within the sentence
    for c1 in vocab:
        for c2 in vocab:
            for c3 in vocab:
                result.add(c1 + c2 + c3)
    # Add trigrams of form c1c2#, last two characters of the sentence
    for c1 in vocab:
        for c2 in vocab:
            result.add(c1 + c2 + "#")
    return result


# Get trigram counts from current line
def get_trigram_counts(line: str):
    if len(line) == 0:
        return dict()
    # Add start and end of sentence tokens to the line
    line = "##" + line + "#"
    counts = dict()
    for i in range(3, len(line)):
        ngram = line[i - 3: i]
        history = ngram[0: 3 - 1]
        current = ngram[3 - 1]
        if history not in counts:
            counts[history] = dict()
        if current not in counts[history]:
            counts[history][current] = 0
        counts[history][current] += 1
    return counts


# Train model with add-one smoothing
def train_model(file: str):
    # Generate all possible trigrams
    vocab = "abcdefghijklmnopqrstuvwxyz0. "
    v = len(vocab)
    all_possible_trigrams = generate_all_trigrams(vocab)

    # Create and pre-populate the data structure holding probabilities
    probabilities = dict()
    for trigram in all_possible_trigrams:
        history = trigram[0:2]
        current = trigram[2]
        if history not in probabilities:
            probabilities[history] = dict()
        if current not in probabilities[history]:
            probabilities[history][current] = 0

    fp = open(file, 'r')
    for line in fp:
        counts = get_trigram_counts(preprocess_line(line))
        for history in counts:
            for current in counts[history]:
                probabilities[history][current] += counts[history][current]

    # Apply add-one smoothing and normalization
    for history in probabilities:
        total_count = sum(probabilities[history].values())
        for current in probabilities[history]:
            probabilities[history][current] += 1
            probabilities[history][current] /= (total_count + v)

    save_model("{}_trigram_model".format(file), probabilities)
    fp.close()
    return probabilities


# Generate specified number of characters from the model
def generate_from_ML(model: dict, n: int):
    history = "##"
    result = ""
    count = 0
    while count < n:
        # List of characters from which next character is chosen
        population = list(model[history].keys())
        # Probabilities corresponding to the the population
        weights = list(model[history].values())
        # Pick random character basing on its probability
        choice = random.choices(population=population, weights=weights, k=1)[0]
        # If the model chooses to start new sentence
        result += choice
        if choice == '#':
            # Reset the history to start a new sentence
            history = "##"
            continue
        count += 1
        history = history[1] + choice
    return result


# Write the model to a text file
def save_model(file: str, model: dict):
    fp = open(file, 'w')
    for hist in model:
        for char in model[hist]:
            line = "{}{}\t{}\n".format(hist, char, model[hist][char])
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



def load_model(file: str):
    fp = open(file, 'r')
    model = dict()
    for line in fp:
        data = line.split("\t")
        hist = data[0][0:2]
        char = data[0][2]
        prob = float(data[1])
        if hist not in model:
            model[hist] = dict()
        if char not in model[hist]:
            model[hist][char] = prob
    return model


def word_length_distribution(file: str):
    fp = open(file, 'r')
    lens_to_counts = dict()
    word_count = 0
    for line in fp:
        line = preprocess_line(line)
        line.replace(".", "")
        line.replace("0", "")
        words = line.split(" ")
        for word in words:
            n = len(word)
            if n == 0:
                continue
            if n not in lens_to_counts:
                lens_to_counts[n] = 0
            lens_to_counts[n] += 1
            word_count += 1
    # normalize the counts
    av_len = sum([k * v for (k, v) in lens_to_counts.items()]) / word_count
    for count in lens_to_counts:
        lens_to_counts[count] /= word_count
    return lens_to_counts, av_len


if __name__ == '__main__':
    filename = sys.argv[1]
    n_generate = int(sys.argv[2])
    language_model = train_model(filename)
    print(generate_from_ML(language_model, n_generate))
    print("\n\n")
    (dist, av) = word_length_distribution(filename)
    print("Average word length for {} is {}".format(filename, av))
    lists = sorted(dist.items())  # sorted by key, return a list of tuples

    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.bar(x, y)
    plt.xlabel("Word length")
    plt.ylabel("Normalized frequency")
    plt.title("Normalized word length frequencies for {}".format(filename))
    plt.show()

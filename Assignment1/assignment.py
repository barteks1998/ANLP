import random
import re
import sys


# Remove any symbols than are not english alphabet characters, space, coma or a digit
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
    for condition in model:
        for character in model[condition]:
            line = "{}{}\t{}\n".format(condition, character, model[condition][character])
            fp.write(line)
    fp.close()


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


if __name__ == '__main__':
    filename = sys.argv[1]
    n_generate = int(sys.argv[2])
    language_model = train_model(filename)
    print(generate_from_ML(language_model, n_generate))
    print(len(generate_from_ML(language_model, n_generate).replace("#", "")))
    print("\n\n")
    print(generate_from_ML(load_model("model-br.en"), n_generate))

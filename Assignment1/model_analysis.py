fp = open("model-br.en", "r")
trigrams = 0
probabilities = dict()
most_often_seen = 0
most_often_prob = None
for line in fp:
    trigrams += 1  # increment the number of observed trigrams
    prob = line[4:-1]
    if prob not in probabilities:
        probabilities[prob] = 0

    probabilities[prob] += 1

    if probabilities[prob] > most_often_seen:
        most_often_seen = probabilities[prob]
        most_often_prob = prob

print("Probability {} has been observed {} times.".format(most_often_prob, most_often_seen))
print("{} different trigrams observed.".format(trigrams))
print("{} different probabilities observed.".format(len(probabilities)))

for prob in probabilities:
    print("\tProbability {} observed {} times".format(prob, probabilities[prob]))


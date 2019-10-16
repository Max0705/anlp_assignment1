import re
import random
import math
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split


# task1
def preprocess_line(str):
    # remove the other characters
    new_str = re.sub('[^a-zA-Z0-9. ]', '', str)

    # convert all digits to 0
    new_str = re.sub('[0-9]', "0", new_str)

    # convert all English characters to lower case
    new_str = new_str.lower()

    # add '##' at the beginning and '#' at the end of each line
    new_str = '##' + new_str + '#'

    new_str = ' '.join(new_str.split())

    return new_str


# Task 3
vocab = [' ', '#', '.', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
         's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


# count n-gram sequence in each line
def count_ngrams(ngram_count, str, n):
    for i in range(0, len(str) - n + 1):
        ngram = str[i:i + n]
        if ngram not in ngram_count:
            ngram_count[ngram] = 1
        else:
            ngram_count[ngram] += 1
    return ngram_count


# count unigrams, bigrams and trigrams of a text and generates trigram probabilities using interpolation
def language_model(input_file, language, lambda1, lambda2, lambda3):
    unigram_count = {}
    bigram_count = {}
    trigram_count = {}
    for line in input_file:
        # process line as described in task1
        line = preprocess_line(line)
        unigram_count = count_ngrams(unigram_count, line, 1)
        bigram_count = count_ngrams(bigram_count, line, 2)
        trigram_count = count_ngrams(trigram_count, line, 3)

    total = sum(unigram_count.values())
    # estimate trigram probabilities
    prob = {}
    for c1 in vocab:
        for c2 in vocab:
            for c3 in vocab:
                # to avoid sequences like '# #' and ' # '
                if c1 == '#' and c3 == '#':
                    continue
                if c1 != '#' and c2 == '#':
                    continue
                seq1 = ''.join([c3])
                seq2 = ''.join([c2, c3])
                seq3 = ''.join([c2])
                seq4 = ''.join([c1, c2, c3])
                seq5 = ''.join([c1, c2])
                if seq1 not in unigram_count:
                    unigram_count[seq1] = 0
                if seq2 not in bigram_count:
                    bigram_count[seq2] = 0
                if seq3 not in unigram_count:
                    unigram_count[seq3] = 0
                if seq4 not in trigram_count:
                    trigram_count[seq4] = 0
                if seq5 not in bigram_count:
                    bigram_count[seq5] = 0

                # interpolation
                prob[seq4] = lambda1 * ((unigram_count[seq1] + 1) / (total + 30)) + lambda2 * (
                        (bigram_count[seq2] + 1) / (unigram_count[seq3] + 30)) + lambda3 * (
                                     (trigram_count[seq4] + 1) / (bigram_count[seq5] + 30))

    # write the trigram model probabilities into file
    output_file = open('trigram_model.' + language, 'w')
    for item in prob:
        # output_file.write(item + '\t' + str(prob[item]) + '\n')
        output_file.write(item + '\t' + '%e' % prob[item] + '\n')

    output_file.close()


# Split train text into 2 parts: a held-out (validation) text and a training text

def split_input_file(input_file):
    #
    text = []
    with open(input_file) as f:
        for line in f:
            line = preprocess_line(line)
            text.append(line)
        validation, training = train_test_split(text, train_size=0.2, random_state=1)

    # save validation text into txt file
    with open("validation", "w") as f:
        for line in validation:
            f.write("".join(line) + "\n")
            # save test text into txt file
    with open("new_training", "w") as f:
        for line in training:
            f.write("".join(line) + "\n")


# Train the training text with different alphas and choose the one that minimizes the perplexity on the validation test
def choose_alpha(train_file, validation_file, language):
    perplexities = dict()
    for lambda1 in np.arange(0.1, 1, 0.1):
        for lambda2 in np.arange(0.1, 1, 0.1):
            if lambda1 + lambda2 >= 1:
                continue
            lambda3 = 1 - lambda1 - lambda2
            training = open(train_file, 'r')
            language_model(training, language, lambda1, lambda2,
                           lambda3)  # generate a model for each value of lambdas in the range
            perplexities[tuple([lambda1, lambda2, lambda3])] = calculate_perplexity('trigram_model.' + language,
                                                                                    validation_file)  # compute perplexity on the validation text
    # Save lambdas that minimizes perplexity
    best_lambda = min(perplexities, key=perplexities.get)
    # best_alpha = list(best_alpha)
    # Plot perplexities
    # plt.plot(list(perplexities.keys()), list(perplexities.values()))
    # plt.plot(best_alpha, perplexities[best_alpha], marker='o')
    # plt.xlabel("alpha")
    # plt.ylabel("Perplexity")
    # plt.show()

    return best_lambda[0], best_lambda[1], best_lambda[2]


# Task 4
N = 300


def generate_from_LM(model_file_name):
    f = open(model_file_name)
    # read the estimated probability
    model = {}
    for line in f:
        line = line.split('\t')
        model[line[0]] = float(line[1])

    output = ''
    head = '##'
    output += head
    # generate the other characters
    while len(output) < N:
        population = [k for (k, v) in model.items() if k.startswith(head)]
        weights = [model[k] for k in population]
        trigram_picked, = random.choices(population=population, weights=weights, k=1)
        # print(trigram_picked)
        ch_picked = trigram_picked[-1]
        output += ch_picked
        head = output[-2:]
        if ch_picked == '#':
            head = '##'
            output += '\n##'

    return output


# Task 5

def calculate_perplexity(model, test_file):
    # read model
    f1 = open(model, 'r')
    prob = {}
    for line in f1:
        line = line.split('\t')
        prob[line[0]] = float(line[1])

    total_logp = 0
    count = 0

    # read test file
    f2 = open(test_file, 'r', encoding="ISO-8859-1")
    for line in f2:
        # process line as described in task1
        line = preprocess_line(line)
        for i in range(0, len(line) - 2):
            p = prob[line[i:i + 3]]
            total_logp += -math.log2(p)
            count += 1

    Hm = total_logp / count
    PPm = 2 ** Hm

    f1.close()
    f2.close()
    return PPm


# task6


if __name__ == '__main__':
    # task3
    split_input_file('./data/training.en')
    input_file = open('./data/training.en', 'r')
    lambda1, lambda2, lambda3 = choose_alpha('new_training', 'validation', 'en')
    language_model(input_file, 'en', lambda1, lambda2, lambda3)
    input_file.close()

    split_input_file('./data/training.es')
    input_file = open('./data/training.es', 'r')
    lambda1, lambda2, lambda3 = choose_alpha('new_training', 'validation', 'es')
    language_model(input_file, 'es', lambda1, lambda2, lambda3)
    input_file.close()

    split_input_file('./data/training.de')
    input_file = open('./data/training.de', 'r')
    lambda1, lambda2, lambda3 = choose_alpha('new_training', 'validation', 'de')
    language_model(input_file, 'de', lambda1, lambda2, lambda3)
    input_file.close()

    # task4
    print('output of model-br.en:')
    print(generate_from_LM('./data/model-br.en'))
    print('output of our English language model:')
    print(generate_from_LM('trigram_model.en'))

    # task5
    print('perplexity on English model:')
    print(calculate_perplexity('trigram_model.en', './data/test'))
    print('perplexity on Spanish model:')
    print(calculate_perplexity('trigram_model.es', './data/test'))
    print('perplexity on German model:')
    print(calculate_perplexity('trigram_model.de', './data/test'))

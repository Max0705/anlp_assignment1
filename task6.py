import re
import random
import math
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import time


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

    # new_str = ' '.join(new_str.split())

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


# count bigrams and trigrams of a text and generates trigram probabilities using add-alpha smoothing
def language_model_2(input_file, language):
    bigram_count = {}
    trigram_count = {}
    for line in input_file:
        # process line as described in task1
        line = preprocess_line(line)
        bigram_count = count_ngrams(bigram_count, line, 1)
        trigram_count = count_ngrams(trigram_count, line, 2)

    # estimate trigram probabilities
    prob = {}
    for c1 in vocab:
        for c2 in vocab:
            seq2 = ''.join([c1])
            seq3 = ''.join([c1, c2])
            if seq2 not in bigram_count:
                bigram_count[seq2] = 0
            if seq3 not in trigram_count:
                trigram_count[seq3] = 0

            # add one smoothing
            prob[seq3] = (trigram_count[seq3] + 1) / (bigram_count[seq2] + 30)

    # write the trigram model probabilities into file
    output_file = open('bigram_model.' + language, 'w')
    for item in prob:
        # output_file.write(item + '\t' + str(prob[item]) + '\n')
        output_file.write(item + '\t' + '%e' % prob[item] + '\n')

    output_file.close()


def language_model_3(input_file, language):
    bigram_count = {}
    trigram_count = {}
    for line in input_file:
        # process line as described in task1
        line = preprocess_line(line)
        bigram_count = count_ngrams(bigram_count, line, 2)
        trigram_count = count_ngrams(trigram_count, line, 3)

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
                seq2 = ''.join([c1, c2])
                seq3 = ''.join([c1, c2, c3])
                if seq2 not in bigram_count:
                    bigram_count[seq2] = 0
                if seq3 not in trigram_count:
                    trigram_count[seq3] = 0

                # add alpha smoothing
                if c1 == '#' and c2 != '#':
                    prob[seq3] = (trigram_count[seq3] + 1) / (bigram_count[seq2] + 29)
                else:
                    prob[seq3] = (trigram_count[seq3] + 1) / (bigram_count[seq2] + 30)

    # write the trigram model probabilities into file
    output_file = open('trigram_model.' + language, 'w')
    for item in prob:
        # output_file.write(item + '\t' + str(prob[item]) + '\n')
        output_file.write(item + '\t' + '%e' % prob[item] + '\n')

    output_file.close()


def language_model_4(input_file, language):
    bigram_count = {}
    trigram_count = {}
    for line in input_file:
        # process line as described in task1
        line = preprocess_line(line)
        bigram_count = count_ngrams(bigram_count, line, 3)
        trigram_count = count_ngrams(trigram_count, line, 4)

    # estimate trigram probabilities
    prob = {}
    for c1 in vocab:
        for c2 in vocab:
            for c3 in vocab:
                for c4 in vocab:
                    if c3 == '#':
                        continue
                    if c1 != '#' and c2 == '#':
                        continue
                    seq2 = ''.join([c1, c2, c3])
                    seq3 = ''.join([c1, c2, c3, c4])
                    if seq2 not in bigram_count:
                        bigram_count[seq2] = 0
                    if seq3 not in trigram_count:
                        trigram_count[seq3] = 0

                    # add one smoothing
                    prob[seq3] = (trigram_count[seq3] + 1) / (bigram_count[seq2] + 30)

    # write the trigram model probabilities into file
    output_file = open('4gram_model.' + language, 'w')
    for item in prob:
        # output_file.write(item + '\t' + str(prob[item]) + '\n')
        output_file.write(item + '\t' + '%e' % prob[item] + '\n')

    output_file.close()


def language_model_5(input_file, language):
    bigram_count = {}
    trigram_count = {}
    for line in input_file:
        # process line as described in task1
        line = preprocess_line(line)
        bigram_count = count_ngrams(bigram_count, line, 4)
        trigram_count = count_ngrams(trigram_count, line, 5)

    # estimate trigram probabilities
    prob = {}
    for c1 in vocab:
        for c2 in vocab:
            for c3 in vocab:
                for c4 in vocab:
                    for c5 in vocab:
                        if c3 == '#' or c4 == '#':
                            continue
                        if c1 != '#' and c2 == '#':
                            continue

                        seq2 = ''.join([c1, c2, c3, c4])
                        seq3 = ''.join([c1, c2, c3, c4, c5])
                        if seq2 not in bigram_count:
                            bigram_count[seq2] = 0
                        if seq3 not in trigram_count:
                            trigram_count[seq3] = 0

                        # add one smoothing
                        prob[seq3] = (trigram_count[seq3] + 1) / (bigram_count[seq2] + 30)

    # write the trigram model probabilities into file
    output_file = open('5gram_model.' + language, 'w')
    for item in prob:
        # output_file.write(item + '\t' + str(prob[item]) + '\n')
        output_file.write(item + '\t' + '%e' % prob[item] + '\n')

    output_file.close()


# Task 5
def calculate_perplexity(model, test_file, n):
    # read model
    f1 = open(model, 'r')
    prob = {}
    linenum = 0
    for line in f1:
        linenum += 1
        line = line.split('\t')
        prob[line[0]] = float(line[1])
    print('line_number:' + str(linenum))
    total_logp = 0
    count = 0

    # read test file
    f2 = open(test_file, 'r', encoding="ISO-8859-1")
    for line in f2:
        # process line as described in task1
        line = preprocess_line(line)
        for i in range(0, len(line) - n + 1):
            p = prob[line[i:i + n]]
            total_logp += -math.log2(p)
            count += 1

    Hm = total_logp / count
    PPm = 2 ** Hm

    f1.close()
    f2.close()
    return (PPm)


# task6


if __name__ == '__main__':
    # bigram
    print('bigram model')
    t1 = time.time()
    input_file = open('./data/training.en', 'r')
    language_model_2(input_file, 'en')
    t2 = time.time()
    print('run_time:' + str(t2 - t1) + 's')
    ppm = calculate_perplexity('bigram_model.en', './data/test', 2)
    print('perplexity:' + str(ppm) + '\n')
    input_file.close()

    # trigram
    print('trigram model')
    t1 = time.time()
    input_file = open('./data/training.en', 'r')
    language_model_3(input_file, 'en')
    t2 = time.time()
    print('run_time:' + str(t2 - t1) + 's')
    ppm = calculate_perplexity('trigram_model.en', './data/test', 3)
    print('perplexity:' + str(ppm) + '\n')
    input_file.close()

    # 4gram
    print('4gram model')
    t1 = time.time()
    input_file = open('./data/training.en', 'r')
    language_model_4(input_file, 'en')
    t2 = time.time()
    print('run_time:' + str(t2 - t1) + 's')
    ppm = calculate_perplexity('4gram_model.en', './data/test', 4)
    print('perplexity:' + str(ppm) + '\n')
    input_file.close()

    # 5gram
    print('5gram model')
    t1 = time.time()
    input_file = open('./data/training.en', 'r')
    language_model_5(input_file, 'en')
    t2 = time.time()
    print('run_time:' + str(t2 - t1) + 's')
    ppm = calculate_perplexity('5gram_model.en', './data/test', 5)
    print('perplexity:' + str(ppm) + '\n')
    input_file.close()

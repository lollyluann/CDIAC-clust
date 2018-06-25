import numpy as np
import sys

def pos_ngram_dist(X,Y,n,i,j):
    dist_sum = 0
    for u in range(1,n + 1):

        # the " - 1" is to specify the i + u th character in the string,
        # since the first character is indexed with 0.
        x_i = X[i + u - 1]
        y_j = Y[j + u - 1]
        unigram_dist = 1
        if (x_i == y_j):
            unigram_dist = 0
        dist_sum = dist_sum + unigram_dist
    return dist_sum / n

#=========1=========2=========3=========4=========5=========6=========7=
# n is the length of the n-grams.
# RETURNS: the n-gram word distance between the strings X and Y
# NOTE that this is not the same thing as n-gram distance. The WORD
# distance means that we are doing normalization by length of the words
# and affixing as well. Affixing is making sure the beginning and ends
# are weighted as highly as the characters in the middle of the strings.
def word_dist(n, str_1,str_2):
    # just making copies of the arguments so we aren't modifying them.
    X = str_1
    Y = str_2
    length_X = len(X)
    length_Y = len(Y)
    # we are affixing, only focusing on the beginning because those
    # characters are empirically more important. We use "|" as a
    # special character because this is built for working with UNIX
    # paths and we guess that it's pretty uncommon to use it in such a
    # context. Note we're adding n - 1 copies of the special string to
    # the beginning because our n-grams have length n. If we added n
    # copies then we would have an n-gram in the beginning that
    # contained none of the original string. Bad juju.
    for u in range(1,n):
        X = "|" + X
        Y = "|" + Y

#=========1=========2=========3=========4=========5=========6=========7=
    # D["X"]["Y"]

    # We initialize the dynamic programming 2-dimensional array for
    # computing distance.
    D = [[0 for y in range(length_Y + 1)] for x in range(length_X + 1)]
    for i in range(0,length_X + 1):
        D[i][0] = i
    for j in range(1,length_Y + 1):
        D[0][j] = j

    # We loops over the array to compute all the recurrence relation
    # values.
    for i in range(1, length_X + 1):
        for j in range(1, length_Y + 1):
            # just exactly the recurrence relation, copied from the
            # n-gram distance paper.
            sub_dist_candidates = [D[i - 1][j] + 1, D[i][j - 1] + 1,
            D[i - 1][j - 1] + pos_ngram_dist(X,Y,n,i - 1, j - 1)]
            D[i][j] = np.amin(sub_dist_candidates)
    # the length of the longer of the two strings X and Y.
    max_length = np.amax([length_X,length_Y])
    return (D[length_X][length_Y] / max_length)

# string_1 = "kylechard"
# string_2 = "aaronelmore"
# dist = word_dist(string_1,string_2,2)
# print(dist)



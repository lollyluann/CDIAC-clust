from collections import Counter

import pandas as pd
import numpy as np

import math
import sys
import os

# THIS FUNCTION PROVIDES AN ALTERNATE DISTANCE METRIC TO SILHOUETTE
# SCORE. 

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: a list of the frequencydrop scores of each of the clusters,
#          and a total, the average of all these scores. 
def compute_freqdrop_score(cluster_directories):

    # list of scores, total score
    freqdrop_scores = []
    freqdrop_scores_scaled = []
    freqdrop_total = 0 

    dataset_size = 0
 
    # iterate over directory, frequency lists 
    for path_counts in cluster_directories:
    
        # get the total number of files in cluster
        cluster_size = 0
        for key, value in path_counts.items():
            cluster_size += value
        dataset_size += cluster_size

    j = 0
    
    # iterate over directory, frequency lists 
    for path_counts in cluster_directories:

        # length of the current dictionary, the number of unique
        # directories in this cluster
        m = len(path_counts)

        freqdrop_score = 0
        freqdrop_scaled = 0
        
        # in the nontrivial case
        if m > 1:
            sigma = 0
            delta = 1

            # get the total number of files in cluster
            cluster_size = 0
            for key, value in path_counts.items():
                cluster_size += value
 
            # Create a dataframe from path_counts        
            df = pd.DataFrame.from_dict(path_counts, orient='index')
            
            # rename the frequency axis
            df = df.rename(columns={ df.columns[0]: "freqs" })
            
            # sort it with highest freqs on top
            sorted_df = df.sort_values("freqs",ascending=False)

            # make a new index, use the directory names as a new column
            sorted_df = sorted_df.reset_index()

            # get just the frequencies column
            freq_df = sorted_df.loc[:,'freqs']

            # list of frequency drop values
            diffs = []

            # list of frequencies in descending order
            freq_list = freq_df.values

            # add all frequency differences to the list
            for i in range(len(freq_list) - 1):
                diff = freq_list[i] - freq_list[i + 1]
                diffs.append(diff)

            # find the largest drop in frequency
            max_diff = 0
            max_diff_index = 0
            for i in range(len(diffs)):
                # print(diffs[i])
                if diffs[i] >= max_diff:
                    max_diff = diffs[i]
                    max_diff_index = i
            # print("max_diff_index: ", max_diff_index)

            # get number of files in the head
            head_size = 0
            for i in range(max_diff_index + 1):
                head_size += freq_list[i]
            # print("head_size: ", head_size)

            if m == 2:
                # print("thinks m = 2. ")
                sigma = 0 
            else:
                # print(" IN else. ")
                delta = max_diff_index + 1
                sigma = math.log(delta, m - 1)
            print("m: ", m)
            print("sigma: ", sigma)
            print("delta: ", delta)
            print("cluster_size: ", cluster_size)
            
            numerator = 1 - sigma
            print("1 - sigma: ", numerator)
            numerator = math.pow(numerator, 2)
            freqdrop_score = (numerator * head_size) / cluster_size
        
        else:
            freqdrop_score = 1
 
        freqdrop_scores.append(freqdrop_score)
        print("Frequency drop score for cluster", 
              j, "is: ", freqdrop_score)
        j += 1
    
        freqdrop_scaled = freqdrop_score * cluster_size / dataset_size
        freqdrop_scores_scaled.append(freqdrop_scaled) 

    freqdrop_total = sum(freqdrop_scores_scaled)
    print("Total frequency drop score: ", freqdrop_total)

    return freqdrop_scores, freqdrop_total

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=


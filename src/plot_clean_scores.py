import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import csv
import sys

# ARGUMENTS
scores_path = sys.argv[1]


def plot_scores(scores_path):
     
    with open(scores_path, 'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',', quotechar='|')
        rows = list(rows)        
        numerical_rows = rows[1:len(rows)]
        scores_array = np.array(numerical_rows)
        
        # delete last column, it's empty
        scores_array = np.delete(scores_array, 4, 1)
        print(scores_array)


    fig = plt.figure(figsize=(5,5))
    plt.plot(scores_array[:,0], scores_array[:,1])

    plt.savefig("scores_plot", dpi=300)

    scores_list = []    
    return scores_list





def main():
    
    scores_list = plot_scores(scores_path)
    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
    main() 

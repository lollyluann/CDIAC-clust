import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import seaborn as sns
import pandas as pd
import numpy as np
sns.palplot(sns.color_palette("cubehelix", 8))

import csv
import sys

#=========1=========2=========3=========4=========5=========6=========7=

# ARGUMENTS
scores_path = sys.argv[1]

#=========1=========2=========3=========4=========5=========6=========7=

def plot_scores(scores_path):
     
    with open(scores_path, 'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',', quotechar='|')
        rows = list(rows)        
        numerical_rows = rows[1:len(rows)]
        scores_array = np.array(numerical_rows)
        
        # delete last column, it's empty
        scores_array = np.delete(scores_array, 4, 1)
        scores = pd.DataFrame(scores_array)
        scores.columns = ['shuffle_ratio', 
                          'freqdrop_score', 
                          'silhouette_score', 
                          'naive_tree_dist_score']
        # scores.set_index(0, inplace=True)
        print(scores)

        # fig, ax = plt.figure(figsize=(5,5))
        fig, ax = plt.subplots(1,1)
        scores = scores.astype(float)
        plt.sca(ax)
        plt.style.use('fivethirtyeight') 
        labels = ['0', '0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1']
        
        plt.xticks(range(11), labels)

        scores.plot(x='shuffle_ratio', ax=ax, use_index=False, legend=False)
        

        # ax.set_xlim([-0.1, 1.1])
        '''
        start, end = ax.get_xlim()
        print("start", start)
        print("end", end)
        # plt.xticks(np.arange(0, 1.1, step=0.1))
        '''

        plt.savefig("scores_plot.svg")

    return 





def main():
    
    plot_scores(scores_path)
    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
    main() 

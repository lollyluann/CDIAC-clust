# paths\_clustering

Information about files and folders in this directory:

## cluster\_ngrams.py
DEPRECATED (only works for small datasets). 
Arguments:
[1] = top level directory of dataset. 
[2] = token length (positive integer). 

Generates n-grams from the file paths, and clusters them using hierarchical clustering, and prints the clusters to a file called `ngram_clusters_output.txt`.

## filename\_token\_clustering.py
Arguments: 
[1] = vectors.txt file, generated from GloVe algorithm. 
[2] = minimum value of k to try for k-means. 
[3] = maximum value of k to try for k-means. 

Runs through all values of k in the specified range and runs k-means clustering on the vectors from the input file. Prints the optimal clustering to a file called `clusters.txt` chosen via silhouette score. 

## clusters.txt
Cluster output from `filename_token_clustering.py`.  

## generate\_token\_dict.py
Arguments:
[1] = top level directory of dataset. 
[2] = token length (positive integer). 

Generates ngram tokens and runs DFS on the directory. Generates a dict that maps filenames to filepaths, and one that maps filenames to tokens. 

## create\_index.py
Arguments:
[1] = top level directory of dataset. 
[2] = token length (positive integer). 

Generates a dictionary that maps tokens to a list of paths which contain that token. Creates inveted index. 

## ngram\_dist.py
Arguments: None. 

Implementation of an n-gram distance metric. 

## plot\_embedding\_3D.py
Arguments:
[1] = word embedding file as a txt document. 
[2] = cluster output as a txt file. 

Projects the vectors down into 3D using PCA, Creates a dict that maps paths to 3D vectors, plots the clusters in 3D, painting each a different color. Outputs to an image called `embedding_3D.png`. 

# embedding\_3D.png
Image genearted from the above script. 

# GloVe
Word embedding generation utilities. 




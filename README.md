# CDIAC-clust

## cluster\_ngrams.py
* cluster\_ngrams: clusters the ngrams
* d1\_ngram\_dist: computes Damerau-Levenshtein distance between ngrams
* get\_tokens: gets a dictionary of files to path tokens
* dict\_vals\_list: gets a list of all ngrams
* main: clusters the ngrams and prints clusters to "ngram\_clusters\_output.txt"

## create\_index.py
* DFS: gets a list of every file path in a directory
* gen\_tokens: gets tokens of some length from the filenames
* main: generates an inverted index mapping ngrams to paths

## generate\_shortened\_token\_dict.py
* DFS: gets a list of every file path in a directory (without the root folder)
* gen\_tokens: gets tokens of some length from the filenames
* get\_all\_paths: writes all path names in a directory to "paths.txt"
* main: maps file names to path tokens and writes to "file\_pathtokens\_dict2.txt" AND maps file names to paths and writes to "file\_path\_dict2.txt"

## generate\_token\_dict.py
* DFS: gets a list of every file path in a directory
* gen\_tokens: gets tokens of some length from the filenames
* main: maps file names to path tokens and writes to "file\_pathtokens\_dict.txt" AND maps file names to paths and writes to "file\_path\_dict.txt"

## ngram\_dist\_clust.py
* get\_paths:
* dict\_vals\_list:
* lol

## ngram\_dist.py
* pos\_ngram\_dist:
* word\_dist:

## plot\_embedding\_3D.py
* build\_word\_vector\_matrix: returns vectors and class labels from vector files
* main: does PCA and plots the embedding in 3 dimensions

## sklearn\_clustering.py
* autovivify\_list: a class that does something related to hashing
* build\_word\_vector\_matrix: returns vectors and class labels from vector files
* find\_word\_clusters: returns the words in each cluster
* main: clusters based on word embeddings and writes all clusters to "clusters.txt"

## GloVe
* creates "vectors.txt" file given all path names


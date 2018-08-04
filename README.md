# CDIAC-clust

Only two scripts from `src` are currently intended to be run: `main.py` and `ensemble_test.py`. 

## main.py

This runs the entire clustering and cleanliness evaluation pipeline. All the possible arguments are in `options.py`. It does not modify, rename, or delete any files in the source dataset. However, it does make copies of some files and place them in a separate directory in order to convert data to a single format. It has the ability to analyze the file extension composition of the dataset, preprocess data for clustering, cluster all tabular and text data, print the results and cluster distributions to a `.pdf`, and compute an estimate of the cleanliness of the dataset. 

Example usage:
`python3 main.py --dataset_path ~/pub8/ --plot_extensions y --convert y --cluster_struct y --cluster_text y  --minibatch_kmeans y --num_clusters_start 5 --num_clusters_end 30`

## ensemble\_test.py

This runs a comparison of the frequency drop score, silhouette score, and naive tree distance cohesion score on the specified dataset. It will print the results to a `.csv` file. It makes a copy of the dataset and progressively shuffles this copy so as to see how well the scores measure the loss in organizational structure. 

## commit\_work.py
First argument: string in quotes for commit message. 
Example: `commit_work.py "updated the grow_apples.py script."`


from options import load_arguments
import converting_utilities
import document_clustering
import schema_clustering
import extensions
import sys
import os

# EVERY OTHER FUNCTION WILL BE CALLED FROM HERE.

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    
    print("ARGUMENTS: ")
    args = load_arguments()
    print("Arguments loaded. ")

    if args.num_clusters_end < args.num_clusters_start:
        args.num_clusters_end = args.num_clusters_start
   
    if args.plot_extensions.lower() == 'y':
        extensions.plot_extensions(args.dataset_path,
                                  args.num_extensions)

    if args.convert.lower() == 'y':
        converting_utilities.convert(args.dataset_path, 
                                     args.num_extensions,
                                     args.num_processes)

    if args.cluster_struct.lower() == 'y':
        start = args.num_clusters_start
        end = args.num_clusters_end
        num_clusters = start
        max_struct_ensemble = 0
        optimal_k = start
        print("Clustering structured files for all k: " + str(start) + "<=k<=" + str(end) + "...\n") 
        while num_clusters <= end:
            struct_ensemble = schema_clustering.runflow(args.dataset_path, 
                                      num_clusters, 
                                      args.overwrite_distmat_struct, 
                                      args.overwrite_plot_struct,
                                      args.fill_threshold)
            print("Schema clustering with k="+str(num_clusters)+" yields ensemble score of " + str(struct_ensemble))
            if struct_ensemble>max_struct_ensemble:
                max_struct_ensemble = struct_ensemble
                optimal_k = num_clusters
            num_clusters += 1
        print("K with highest ensemble score:", str(optimal_k))

    
    if args.cluster_text.lower() == 'y':
        start = args.num_clusters_start
        end = args.num_clusters_end
        num_clusters = start
        max_text_ensemble = [0,num_clusters]
        print("Clustering text files for all k: " + str(start) + "<=k<=" + str(end) + "...\n")
        while num_clusters <= end:
            text_ensemble = document_clustering.runflow(num_clusters, 
                                        args.overwrite_tokens_text,
                                        args.overwrite_clusters_text,
                                        args.dataset_path,
                                        args.minibatch_kmeans,
                                        args.num_processes)
            print("Text clustering with k="+str(num_clusters)+" yields ensemble score of " + str(text_ensemble))
            if text_ensemble>max_text_ensemble:
                max_text_ensemble = [text_ensemble, num_clusters]
            num_clusters += 1
        print("K with highest ensemble score:", str(max_text_ensemble[1]))
 
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()

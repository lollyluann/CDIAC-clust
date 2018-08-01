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
        max_struct_score = 0
        optimal_num_clusters = start
        print("Clustering structured files for all k: " + str(start) + "<=k<=" + str(end) + "...\n") 
        while num_clusters <= end:
            scores = schema_clustering.runflow(args.dataset_path, 
                                      num_clusters, 
                                      args.overwrite_distmat_struct, 
                                      args.overwrite_plot_struct,
                                      args.fill_threshold)
            struct_score = scores[0]
            print("Schema clustering with k="+str(num_clusters)+" yields freqdrop score of " + str(struct_score))
            if struct_score>max_struct_score:
                max_struct_score = struct_score
                optimal_num_clusters = num_clusters
            num_clusters += 1
        print("k with highest freqdrop score:", str(optimal_num_clusters))

    
    if args.cluster_text.lower() == 'y':
        start = args.num_clusters_start
        end = args.num_clusters_end
        num_clusters = start
        max_text_score = 0
        optimal_num_clusters = start
        allscores = []
        retokenize = args.overwrite_tokens_text
        print("Clustering text files for all k: " + str(start) + "<=k<=" + str(end) + "...\n")
        while num_clusters <= end:
            if num_clusters > start:
                retokenize = "n"
            sil, frqdrop, text_score = document_clustering.runflow(num_clusters, 
                                        args.overwrite_tokens_text,
                                        args.overwrite_clusters_text,
                                        args.dataset_path,
                                        args.minibatch_kmeans,
                                        args.num_processes)
            print("Text clustering with k="+str(num_clusters)+" yields freqdrop score of " + str(frqdrop))
            allscores.append(frqdrop)
            if frqdrop > max_text_score:
                max_text_score = frqdrop
                optimal_num_clusters = num_clusters
            num_clusters += 1
        for x in range(len(allscores)):
            print("k=" + str(start+x) + " cleanliness=" + str(allscores[x]))
        print("k with highest cleanliness score:", str(optimal_num_clusters))
 
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()

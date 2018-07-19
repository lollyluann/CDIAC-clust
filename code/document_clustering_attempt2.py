import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd
import silhouette
from time import time
from tqdm import tqdm
from glob import glob
from unipath import Path
from six import string_types
import nltk, re, os, codecs, mpld3, sys, random
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from nltk.stem.snowball import SnowballStemmer
from mpl_toolkits.mplot3d import Axes3D

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a string with a directory
    RETURNS: only the last folder name '''
def get_end_directory(directory):
    i = 0
    ct = 0
    for ch in directory[::-1]:
        if ch=="/":
            ct+=1
            if ct>1:
                break
        i += 1
    return directory[len(directory)-i:len(directory)-1].replace("/","")

''' PARAMETER: a full path
    RETURNS: only the filename '''
def get_fname_from_path(path):    
    filename = ""
    for c in path[::-1]:
        if c=="/" or c=="@":
            break
        filename = c+filename
    return filename

''' PARAMETER: a single filename 
    RETURNS: the file without its extension '''
def remove_extension(filename):    
    length = len(filename)
    for ch in filename[::-1]:
        if ch==".":
            break
        length = length - 1 
    return filename[:length-1]

''' PARAMETER: a single string
    RETURNS: the string with all @ replaced with / '''
def str_decode(string):
    return string.replace("@","/")

def get_last_dir_from_path(path):    
    filename = ""
    if path[len(path) - 1] == "/" or path[len(path) - 1] == "@":
        path = path[0:len(path) - 1]
    for c in path[::-1]:
        if c=="/" or c=="@":
            break
        filename = c+filename
    return filename

''' PARAMETER: a directory path
    RETURNS: a list of the immediate children of that directory '''
def get_immediate_subdirectories(a_dir):
    sub_list = []
    for name in os.listdir(a_dir):
        if os.path.isdir(os.path.join(a_dir, name)):
            sub_list.append(os.path.join(a_dir, name)+"/")
    return sub_list

''' PARAM: "directory" - a parent directory containing /pdf/ /doc/ etc
           "dataset_dir" - a directory leading to the dataset
    RETURNS: a list of filenames and a list of the contents of the files 
    DOES: gets all the filenames and their contents of a directory '''   
def get_document_contents(directory, dataset_dir):
    p = Path(os.getcwd()).parent
    file_place = os.path.join(p, "paths_work/")    
    root_name = get_last_dir_from_path(dataset_dir)
    ext_dict_file_loc = os.path.join(file_place,"extension_index_" + root_name + ".npy")
    ext_paths = np.load(ext_dict_file_loc).item()

    filenames = []
    data = []
    i = 1
        
    # a list of the paths of the txt files still in pub8 
    txt_paths = ext_paths.get("txt")
    print("Getting .txt contents from " + dataset_dir)
    for path in tqdm(txt_paths):
        if os.path.isfile(path):
            i = i+1
            # add the filename to "filenames" 
            #filenames.append(get_fname_from_path(path))
            filenames.append(path)
            # read the contents of the file and remove newlines
            fread = open(path, "r", errors='backslashreplace')
            contents = fread.read()
            fread.close()
            contents = contents.replace("\n","")
            # add the string of the contents of the file to "data"
            data.append(contents)
    
    # for every file in the directory of converted files
    conv_folders = get_immediate_subdirectories(directory)
    for folder in conv_folders:
        filetype = get_end_directory(folder)
        if filetype in ["pdf", "doc", "docx"]: #, "xml", "html"]:
            print("Getting ."+folder+" contents")
            for filename in tqdm(os.listdir(folder)):
                cur_file = os.path.join(folder,filename)
                if os.path.isfile(cur_file):
                    i = i + 1
                    # add the non-converted filename to "filenames" 
                    new_name = str_decode(remove_extension(filename))
                    filenames.append(new_name)
                    # read the contents of the file and remove newlines
                    fread = open(cur_file, "r", errors='backslashreplace')
                    contents = fread.read()
                    fread.close()
                    contents = contents.replace("\n","")
                    # add the string of the file contents to "data"
                    data.append(contents)
    
    print("num total files: ", i)
    print("All directory contents retrieved")
    return filenames, data

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: the text of a document
    RETURN: list of stems
    DOES: splits a document into a list of tokens & stems each token '''
def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english")
    # tokenize by sentence, then by word so punctuation is its own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out tokens without letters (e.g., numbers, punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if len(token)>1:
               filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

''' PARAMETER: the text of a document
    RETURN: a list of filtered tokens
    DOES: tokenizes the document only (doesn't stem) '''
def tokenize_only(text):
    # tokenize by sentence, then by word so punctuation is its own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out tokens without letters (e.g., numbers, punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if len(token)>1:
                filtered_tokens.append(token)
    return filtered_tokens

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETERS: "num_clusters" - the number of clusters to cluster into
                "retokenize" - 1 if to retokenize, 0 to not
                "corpusdir" - the directory of converted files
                "dataset_dir" - directory of original dataset
                "n_words" - number of words/cluster to print out '''
def main_function(num_clusters, retokenize, corpusdir, dataset_dir, n_words):
    # gets the filenames and their contents
    # "fnames" is a list of the original location of every file in the
    # source dataset
    # "dataset" is a list of strings. Each string is every word in the 
    # document separated by spaces. 
    fnames, dataset = get_document_contents(corpusdir, dataset_dir)
    #stopwords = nltk.download('stopwords')

    #nltk.download('punkt')
    stemmer = SnowballStemmer("english")

    root_name = get_last_dir_from_path(dataset_dir)
    
    #=========1=========2=========3=========4=========5=========6=======

    if retokenize == "1": 
        print("Tokenizing", len(dataset), "documents...")
        totalvocab_stemmed = []
        totalvocab_tokenized = []
        for i in tqdm(dataset):
            # for each item in the dataset, tokenize/stem
            allwords_stemmed = tokenize_and_stem(i)
            # extend "totalvocab_stemmed" 
            totalvocab_stemmed.extend(allwords_stemmed)
            
            # for each item in the dataset, tokenize only
            allwords_tokenized = tokenize_only(i)
            totalvocab_tokenized.extend(allwords_tokenized)

        # create vocab_frame with "totalvocab_stemmed" or "totalvocab_tokenized"
        vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, 
                    index = totalvocab_stemmed)
        vocab_frame.to_pickle("vocab_frame_" + root_name + ".pkl")
        print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

        #define vectorizer parameters #############0.8############0.2#############0.68##0.12###########################################################################################
        tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                              min_df=1, stop_words='english', use_idf=True, 
                              tokenizer=tokenize_and_stem, ngram_range=(1,3))

        #fits the vectorizer to the dataset
        print("Fitting vectorizer to data...")
        tfidf_matrix = tfidf_vectorizer.fit_transform(dataset) 
        terms = tfidf_vectorizer.get_feature_names()
        np.save("terms_" + root_name + ".npy", terms)
        dist = 1 - cosine_similarity(tfidf_matrix)
        np.save("distance_matrix_" + root_name + ".npy", dist)
        print("Vectorizer fitted to data")
       
        #=========1=========2=========3=========4=========5=========6======

        # cluster using KMeans on the tfidf matrix
        print("Clustering using kmeans with k = " + str(num_clusters) + "...")
        km = KMeans(n_clusters=num_clusters)
        km.fit(tfidf_matrix)
        clusters = km.labels_.tolist()
        print("Kmeans clustering complete")

        # pickle the model, reload the model/reassign the labels as the clusters
        joblib.dump(km, 'doc_cluster_' + root_name + '.pkl')


    km = joblib.load('doc_cluster_' + root_name + '.pkl')
    vocab_frame = pd.read_pickle("vocab_frame_" + root_name + ".pkl")
    terms = np.load("terms_" + root_name + ".npy")
    dist = np.load("distance_matrix_" + root_name + ".npy")
    print("Loaded in existing cluster profile...\n")

    clusters = km.labels_.tolist()
    # get the actual number of clusters in the dataframe, in case some
    # are omitted for some reason
    cluster_list = list(set(clusters))    
    true_num_clusters = len(cluster_list)
    print("number of distinct clusters: ", true_num_clusters)
    print("length of fnames: ", len(fnames))
    print("length of dataset: ", len(dataset))
    print("clusters: ")
    print(clusters)
    distinct_cluster_labels = []
    for label in clusters:
        if label not in distinct_cluster_labels:
            distinct_cluster_labels.append(label)
    

    # create a dictionary "db" of filenames, contents, and clusters
    db = {'filename': fnames, 'content': dataset, 'cluster': clusters}
    # convert "db" to a pandas dataframe
    frame = pd.DataFrame(db, index=[clusters], columns=['filename','cluster'])
    # print the number of files in each cluster
    #print("Number of files in each cluster: ")
    #print(frame['cluster'].value_counts())

    #=========1=========2=========3=========4=========5=========6=========
   
    # open file writer for result output
    output_path = "results_" + root_name + "/"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    os.chdir(output_path)
    fwriter = open("doc_clusters.txt", "w")
    fwriter.write("clusters from text files in: " + corpusdir)

    fwriter.write("\nTop terms per cluster: \n\n")
    print("Top terms per cluster: \n")

    #sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
    
    #print("Printing the index. ")
    #print(frame.index)
    #print(vocab_frame.to_string())   
    print("true number of clusters: ", true_num_clusters)


    all_cluster_words = {}
    # for each cluster
    for i in tqdm(distinct_cluster_labels):
        # we catch key errors because some of the clusters may 
        # not exist. 
        fwriter.write("Cluster " + str(i) + " words: ")
        print("Cluster %d words:" % i, end='')
        
        cluster_words = []
        # print the first "n_words" words in a cluster
        # NO PRINT FOR NOW REMOVE
        ''' 
        for ind in order_centroids[i, : n_words]:
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0],
                    end=",")
            #fwriter.write(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].rstrip('\n') + ", ")
            cluster_words.append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0])
        print()
        #fwriter.write("\n")
          

        all_cluster_words.update({i:cluster_words})

        # NO PRINT FOR NOW REMOVE
        
        # print out the filenames in the cluster
        print("Cluster %d filenames:" % i, end='')
        fwriter.write("Cluster " + str(i) + " filenames: ")
        for filename in frame.loc[i]['filename'].values.tolist():
            print(' %s,' % filename, end='')
            fwriter.write(filename.rstrip('\n') + ", ")
        print("\n")
        fwriter.write("\n\n")
        '''

    fwriter.close()
    print("output written to \"doc_clusters.txt\" in \"results_" + root_name + "\"")

    #=========1=========2=========3=========4=========5=========6========
    '''
    # multidimensional scaling to convert distance matrix into 3 dimensions
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    # set up plot
    fig = plt.figure(figsize=(17,9))
    ax = Axes3D(fig)

    # create data frame with MDS results, cluster numbers, and filenames
    df = pd.DataFrame(dict(x=xs, y=ys, z=zs, label=clusters, filename=fnames)) 
    # group by cluster
    groups = df.groupby('label')

    # for each cluster, plot the files in that cluster
    print("Plotting scatterplot of cluster points...")
    for name, group in tqdm(groups):
        # color = ('#%06X' % random.randint(0,256**3-1))
        color = np.random.rand(3,)
        for t in range(group.shape[0]):
            ax.scatter(group.x.iloc[t], group.y.iloc[t], group.z.iloc[t], 
                c=color, marker='o')
            ax.set_aspect('auto')

    plt.savefig("3D_document_cluster", dpi=300)
    print("scatter plot written to \"3D_document_cluster.png\"")
    '''       
    return frame, all_cluster_words, distinct_cluster_labels

''' PARAMETERS: "frame" - a dataframe containing which files are in which clusters
                "num_clusters" - the number of clusters
                "home_dir" - the parent directory of the dataset (e.g. /home/ljung/)
                "dataset_dir" - the directory of the dataset (e.g. /home/ljung/pub8/)
    DOES: plots a pdf with all the bar charts on it
          each bar chart shows which directories the files in a cluster come from '''
def bar_clusters(frame, distinct_cluster_labels, home_dir, dataset_dir):
    plt.figure("bar")
  
    matplotlib.rcParams.update({'font.size': 4})
    root_name = get_last_dir_from_path(dataset_dir)
    
    print("\nGenerating barcharts...")    
    pdf = matplotlib.backends.backend_pdf.PdfPages("text_barcharts.pdf")
    cluster_directories = []
    # for each cluster, generate a bar chart 
    for i in tqdm(distinct_cluster_labels):
        # We allow for the case where some clusters are missing 
        plt.clf()
        paths_in_cluster = {}
        # get the files associated with the current cluster
        cluster_files = frame.loc[frame['cluster']==i]
        count = 0 
        for index, row in cluster_files.iterrows():
            if count>1:
                path = get_dir_from_path(row['filename'])
                # if the path is already in the cluster, add to count
                if path in paths_in_cluster:
                    paths_in_cluster.update({path:paths_in_cluster.get(path)+1})
                else:
                    paths_in_cluster.update({path:1})
            count+=1
        cluster_directories.append(paths_in_cluster)
        sorted_names = []
        sorted_counts = []
        # sort the paths in ascending order based on # of occurrences
        for e in sorted(paths_in_cluster, key=paths_in_cluster.get, reverse=True):
            trimmed_name = e[len(home_dir):]
            sorted_names.append(trimmed_name)
            sorted_counts.append(paths_in_cluster[e])

        cluster_stats = get_cluster_stats(paths_in_cluster)

        y_pos = np.arange(len(sorted_names))
        plt.bar(y_pos, sorted_counts, align='center', alpha=0.5)
        plt.xticks(y_pos, sorted_names, rotation=90)
        plt.rc('xtick')
        plt.ylabel('Number of files')
        plt.xlabel('Directory')
        plt.title('Directories in Cluster ' + str(i) + "\n" + cluster_stats)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.38, top=0.87)
        
        pdf.savefig(plt.gcf())
    pdf.close()
    p = Path(os.getcwd())
    p2 = Path(p.parent)
    os.chdir(os.path.join(p2.parent,"paths_work"))
    cluster_dir_saved_path = os.path.join(p2.parent, "paths_work/", "cluster_directories_" + root_name + ".npy")
    np.save("cluster_directories_" + root_name + ".npy",cluster_directories)
    return cluster_dir_saved_path

def print_cluster_stats(top_words, frame, dataset_dir, clust_dir_location):
    root_name = get_last_dir_from_path(dataset_dir)
    cluster_directories = np.load(clust_dir_location)
    p = Path(os.getcwd()).parent
    p2 = os.path.join(p, "text_clustering/", "results_" + root_name + "/")
    os.chdir(p2)
    fr = open("cluster_stats.txt", "w")
    total_silhouette, scores = silhouette.compute_silhouette(cluster_directories, "/home/ljung/pub8/")
    num_files_per_cluster = frame['cluster'].value_counts().sort_index().tolist()
    for clust_num in range(len(cluster_directories)):
        c_stats = "Cluster " + str(clust_num) + "\n" + str(num_files_per_cluster[clust_num]) + " files \n"
        c_stats = c_stats + get_cluster_stats(cluster_directories[clust_num])
        c_stats = c_stats + "\nSilhouette score: " + str(scores[clust_num]) #+ "\nTop 10 words: " + ", ".join(top_words.get(clust_num))
        fr.write(c_stats + "\n\n")
    fr.write("\nTotal silhouette score: " + str(total_silhouette))
    fr.close()

''' PARAMETER: a full path
    RETURNS: the path without the filename '''
def get_dir_from_path(path):   
    ind = 0
    for c in path[::-1]:
        ind = ind+1
        if c=="/" or c=="@":
            break
    return path[:len(path)-ind+1]

''' PARAMETER: a list of filepaths
    RETURNS: nearest shared parent directory of all the files given '''
def nearest_shared_parent(filepaths):
    path1 = filepaths[0]
    path1_folders = path1.split("/")
    shared_index = len(path1_folders)
    for i in range(1,len(filepaths)):
        path2 = filepaths[i]
        path2_folders = path2.split("/")
        min_folderlist_len = min(len(path1_folders), len(path2_folders))
        for j in range(min_folderlist_len):
            # if the folder matches
            if path1_folders[j]==path2_folders[j]:
                # save the shared path
                shared_index = j+1
            # once they are no longer equal, stop iterating. 
            else:
                break
    path1 = path1_folders[:shared_index]
    return path1

''' PARAMETER: a dictionary of directories and their counts from ONE cluster
    DOES: returns a string of mean, std-dev, and nearest shared parent for that cluster '''
def get_cluster_stats(one_cluster_directories):
    dir_counts = np.array(list(one_cluster_directories.values()))
    unique = str(len(one_cluster_directories)) + " unique directories"
    avg = "Avg dir frequency: " + str( np.mean(dir_counts))
    med = "Median dir frequency: " + str( np.median(dir_counts))
    std = "Std-dev dir frequency: " + str( np.std(dir_counts))
    nsd = "Nearest shared directory: " + "/".join(nearest_shared_parent(list(one_cluster_directories.keys())))
    return unique+"\n"+avg+"\n"+med+"\n"+std+"\n"+nsd

# MAIN PROGRAM

def main():
    num_clusters = int(sys.argv[1])
    retokenize = sys.argv[2]
    # the directory containing the files not in pub8 you want to cluster
    corpusdir = sys.argv[3] #eg. /home/ljung/converted/
    dataset_dir = sys.argv[4]
   
    # record initial time that program started
    t0 = time()
    fr, all_cluster_words, distinct_cluster_labels = main_function(num_clusters, retokenize, corpusdir, dataset_dir, 10)
    location_cluster_dir = bar_clusters(fr, distinct_cluster_labels, "/home/ljung/", dataset_dir)    
    print_cluster_stats(all_cluster_words, fr, dataset_dir, location_cluster_dir)
    # print total time taken to run program
    print("time taken: ", time()-t0, " seconds")

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf

import numpy as np
import pandas as pd
import time
import queue
from time import time
from tqdm import tqdm
from glob import glob
from unipath import Path
from six import string_types
import multiprocessing as mp
import multiprocessing.queues
from multiprocessing import Pool
import nltk, re, os, codecs, mpld3, sys, random
from nltk.stem.snowball import SnowballStemmer

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D

import frequencydrop
import silhouette
import path_utilities

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAM: "directory" - a parent directory containing pdf/ doc/ etc
           "dataset_path" - a directory leading to the dataset
    RETURNS: a list of filenames and a list of the contents of the files 
    DOES: gets all the filenames and their contents of a directory '''   
def get_document_contents(directory, dataset_path):
    dataset_name, file_place = initialize_output_location(dataset_path)
    
    # load in the extension index file from the output folder
    ext_dict_file_loc = os.path.join(file_place,"extension_index_" + dataset_name + ".npy") 
    ext_paths = np.load(ext_dict_file_loc).item()
    
    # "filenames" list of the paths of files
    # "data" list of the contents of files 
    filenames = []
    data = []
    i = 1
    
    # get contents of txt files still in original dataset
    txt_paths = ext_paths.get("txt")
    #txt_paths.extend(ext_paths.get("py"))
    print("Getting .txt contents from " + dataset_path)
    for path in tqdm(txt_paths):
        if os.path.isfile(path):

            i = i+1
            # add the path of the file to "filenames" 
            filenames.append(path)

            # read the contents of the file and remove newlines
            fread = open(path, "r", errors='backslashreplace')
            contents = fread.read()
            fread.close()
            contents = contents.replace("\n","")
            # add the string of the contents of the file to "data"
            data.append(contents)
    
    # get contents of converted files in the other directory
    conv_folders = path_utilities.get_immediate_subdirectories(directory)
    # for each folder in the directory (e.g. pdf/ doc/)
    for folder in conv_folders:
        filetype = path_utilities.get_last_dir_from_path(folder)
        if filetype in ["pdf", "doc"]:#, "docx", "html", "htm"]: #, "xml"]:
            print("Getting ."+folder+" contents")
            for filename in tqdm(os.listdir(folder)):
                cur_file = os.path.join(folder,filename)
                if os.path.isfile(cur_file):
                    i = i + 1
                    # add the non-converted filename to "filenames" 
                    new_name = path_utilities.str_decode(path_utilities.remove_extension(filename))
                    filenames.append(new_name)

                    # read the contents of the file and remove newlines
                    fread = open(cur_file, "r", errors='backslashreplace')
                    contents = fread.read()
                    fread.close()
                    contents = contents.replace("\n","")
                    # add the string of the file contents to "data"
                    data.append(contents)
    
    print("Num total files: ", i)
    print("All directory contents retrieved")
    return filenames, data

#=========1=========2=========3=========4=========5=========6=========7=

def tokenize_action(text):
    stemmer = SnowballStemmer("english")
    # tokenize by sentence, then by word so punctuation is its own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)] 
    
    # filter out tokens without letters (e.g., numbers, punctuation)
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            # removes tokens with length 1
            if len(token)>1:
               filtered_tokens.append(token)
    
    # stems the tokens
    stems = [stemmer.stem(t) for t in filtered_tokens]    
    return [filtered_tokens, stems]

#=========1=========2=========3=========4=========5=========6=========7=

# ARGUMENTS: "dataset"  list of all document text strings
def tokenize_action_par(dataset, input_queue, output_queue):

    print("Starting a process. ") 
    sys.stdout.flush()

    # for each document
    while True:
        try:
            iteration = input_queue.get()
        except Queue.Empty:
            break
        print("Iteration: ", iteration)

        stemmer = SnowballStemmer("english")
        
        # tokenize by sentence, then by word so punctuation is its own token, only for a single doc at a time
        tokens = [word for sent in nltk.sent_tokenize(dataset[iteration]) for word in nltk.word_tokenize(sent)] 
        
        # filter out tokens without letters (e.g., numbers, punctuation)
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
        
                # removes tokens with length 1
                if len(token)>1:
                   filtered_tokens.append(token)
        
        # stems the tokens
        stems = [stemmer.stem(t) for t in filtered_tokens]

        sys.stdout.flush()

        output_queue.put([[filtered_tokens, stems], iteration])


    print("Ending a process. ")
    sys.stdout.flush() 
    return

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: the text of a document
    RETURN: list of filtered tokens and a list of stems
    DOES: splits a document into a list of tokens & stems each token '''
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    # filter out tokens without letters (e.g., numbers, punctuation)
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            # removes tokens with length 1
            if len(token)>1:
               filtered_tokens.append(token)
    
    # stems the tokens
    stems = [stemmer.stem(t) for t in filtered_tokens]    
    return filtered_tokens, stems

''' PARAMETER: the text of a document
    RETURN: a list of stems
    DOES: for use with Tfidf Vectorizer '''
def tokenize_and_stem_call(text):
    return tokenize_action(text)[1] 

''' PARAMETER: the path leading to the dataset
    RETURNS: the name of the dataset and the output path '''
def initialize_output_location(dataset_path):
    # setup output path as "file_place" outside the repo
    p = Path(Path(os.getcwd()).parent).parent
    dataset_name = path_utilities.get_last_dir_from_path(dataset_path)
    file_place = os.path.join(p, "cluster-datalake-outputs/", dataset_name + "--output")    
    
    if not os.path.isdir(path_utilities.remove_path_end(file_place)):
        os.mkdir(path_utilities.remove_path_end(file_place)) 

    if not os.path.isdir(file_place):
        os.mkdir(file_place)
    return dataset_name, file_place
    
#=========1=========2=========3=========4=========5=========6=========7=

def mkproc(func, arguments):
    p = mp.Process(target=func, args=arguments)
    p.start()
    return p

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETERS: parameters explained in main_function
    RETURNS: a list of filenames and a list of file data
    DOES: retokenizes and catches errors '''
def to_retokenize(retokenize, corpusdir, dataset_path, num_processes):
    trot = time() 
    dataset_name, file_place = initialize_output_location(dataset_path)
    
    # "fnames" is a list of the paths of each file in the  source dataset
    # "dataset" is a list of strings, each containing a document's text
    fnames, dataset = get_document_contents(corpusdir, dataset_path)
    
    # if trying to skip tokenization, check for dependencies 
    if retokenize == "0":
        print("\nAttempting to bypass tokenizing...")
        vf = os.path.join(file_place, "vocab_frame_" + dataset_name + ".pkl")
        tms = os.path.join(file_place, "terms_" + dataset_name + ".npy")
        dm = os.path.join(file_place, "distance_matrix_" + dataset_name + ".npy")
        tf = os.path.join(file_place, "tfidf_matrix_" + dataset_name + ".npy")

        # if any dependency is missing, set to retokenize
        if not os.path.isfile(vf) or not os.path.isfile(tms) or not os.path.isfile(dm) or not os.path.isfile(tf):
            print("One or more dependencies is missing... \nTokenizing...")
            retokenize = "1"
        else:
            print("Success!")
            
    if retokenize == "1":     
        print("\nTokenizing", len(dataset), "documents...")
        data_part = (sys.getsizeof(dataset)/1000)//10
        ind = 0
        tokens_and_stems = []
        num_docs = len(dataset)
        
        while ind<len(dataset):
            incr = int((len(dataset)/data_part)+1) 
            dataset_p = dataset[ind:ind+incr]
            ind += incr

    #===================================================================        
    #   PARALLELIZING TOKENIZATION
    #===================================================================        

        # we instantiate the queue
        input_queue = mp.Queue()
        output_queue = mp.Queue()

        # So we need each Process to take from an input queue, and to
        # output to an output queue. All batch generation prcoesses
        # will read from the same input queue, and what they will be
        # reading is just an integer which corresponds to an iteration
        for iteration in tqdm(range(num_docs)):
            input_queue.put(iteration)

        # CREATE MATRIXMULT PROCESSES
        tokenize_args = (dataset, input_queue, output_queue)
        allprocs = [mkproc(tokenize_action_par, tokenize_args) for x in range(num_processes)]

        # get the tokens and stems from each document out of 
        # "output_queue" and sort them in a list
        # tokens_stems_list is a list of tuples, one for each document
        # these tuples look like ((tokens_for_doc, stems_for_doc), iteration) where
        # "tokens_for_doc" and "stems_for_doc" are lists of the tokens and stems for
        # that document, and "iteration" is the index of the document
        # in the dataset. 
        # CAUTION: they may not be returned in order, iteration not 
        # strictly increasing
        tokens_stems_list = []
        while len(tokens_stems_list) < num_docs:
            tokens_stems_list.append(output_queue.get())

        # join the processes, i.e. end them
        for process in allprocs:
            process.terminate()

        # join the processes, i.e. end them
        for process in allprocs:
            process.join()
        
        # we sort "tokens_stems_list" so "iteration" is strictly
        # increasing
        sorted_tokens_stems_2tuple = sorted(tokens_stems_list, key=lambda x: x[1]) 
        
        # lists of tokens and stems for whole dataset
        tokens = []
        stems = []

        # for each tuple ((tokens_for_doc, stems_for_doc),iteration)
        for two_tuple in sorted_tokens_stems_2tuple:

            # get the tuple (tokens_for_doc,stems_for_doc)
            tokens_stems_for_doc = two_tuple[0] 
            tokens_for_doc = tokens_stems_for_doc[0]
            stems_for_doc = tokens_stems_for_doc[1]

            # extend the list of tokens and stems for whole dataset
            tokens.extend(tokens_for_doc)
            stems.extend(stems_for_doc)

        print("We're done. ")

    #===================================================================        
    #   DONE
    #===================================================================        

        totalvocab_stemmed = tokens
        totalvocab_tokenized = stems

        ''''
        for i in tqdm(dataset):
            # for each item in the dataset, tokenize and stem
            allwords_tokenized, allwords_stemmed = tokenize_and_stem(i)
            with Pool(num_processes) as p:
            # extend "totalvocab_stemmed" and "totalvocab_tokenized"
            totalvocab_stemmed.extend(allwords_stemmed) 
            totalvocab_tokenized.extend(allwords_tokenized)
        '''

        # vocab_frame contains all tokens mapped to their stemmed counterparts
        vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, 
                                   index = totalvocab_stemmed)
        vocab_frame.to_pickle(os.path.join(file_place, "vocab_frame_" + dataset_name + ".pkl"))
        print('There are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
   
        print("Time to tokenize:", time()-trot)    
     
        tfit = time()
        #define vectorizer parameters
        tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                              min_df=1, stop_words='english', use_idf=True, 
                              tokenizer=tokenize_and_stem_call, ngram_range=(1,3))
        
        #fits the vectorizer to the dataset
        print("\nFitting vectorizer to data...")
        tfidf_matrix = tfidf_vectorizer.fit_transform(dataset) 
        np.save(os.path.join(file_place, "tfidf_matrix_" + dataset_name + ".npy"), tfidf_matrix)
        
        # the list of all feature names (tokens)
        terms = tfidf_vectorizer.get_feature_names()
        np.save(os.path.join(file_place, "terms_" + dataset_name + ".npy"), terms)
        
        # distance matrix from the tfidf matrix
        dist = 1 - cosine_similarity(tfidf_matrix)
        np.save(os.path.join(file_place, "distance_matrix_" + dataset_name + ".npy"), dist)
        print("Vectorizer fitted to data")
        print("Time to fit tokenizer:", time() - tfit)

    return fnames, dataset
      
#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETERS: parameters explained in main_function
    DOES: reclusters and catches errors '''
def to_recluster(num_clusters, retokenize, recluster, tfidf_matrix, dataset_path, minibatch):
    dataset_name, file_place = initialize_output_location(dataset_path)
    trailer_text = dataset_name + "_k=" + str(num_clusters)
   
    # if try to bypass clustering, check for dependencies
    if recluster == "0":
        print("\nAttempting to bypass clustering...") 
        dc = os.path.join(file_place, "doc_cluster_" + trailer_text + ".pkl")
        
        # if dependency is missing, set to cluster
        if not os.path.isfile(dc):
            print("\"doc_cluster_" + trailer_text + ".pkl\" is missing... \nClustering...")
            recluster = "1"
        else:
            print("Success!")

        # if retokenize and recluster are 0, only regenerates graphs
        if retokenize == "0":
            print("\nNote: Running without overwriting tokens or clusters only regenerates graphs and text output.")
    
    # if retokenizing, you must recluster
    if retokenize == "1":
        recluster = "1"

    if recluster == "1":
        clustert0 = time()
        
        # cluster using KMeans on the tfidf matrix
        if minibatch == "1":
            km = MiniBatchKMeans(n_clusters=num_clusters)
            print("\nClustering using minibatch kmeans with k = " + str(num_clusters) + "...")
        else:
            km = KMeans(n_clusters=num_clusters, n_jobs=-1)
            print("\nClustering using kmeans with k = " + str(num_clusters) + "...") 
        
        km.fit(tfidf_matrix)
        print("Kmeans clustering complete")
        print("Time to cluster:", time()-clustert0)    
    
        # pickle the model, reload the model/reassign the labels as the clusters
        joblib.dump(km, os.path.join(file_place, 'doc_cluster_' + trailer_text + '.pkl'))

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETERS: "num_clusters" - the number of clusters to cluster into
                "retokenize" - 1 if to retokenize, 0 to not
                "corpusdir" - the directory of converted files
                "dataset_path" - directory of original dataset
                "n_words" - number of words/cluster to print out 
    RETURNS: "frame" - dataframe containing clusters and file paths
             "all_cluster_words" - list of lists of top words in cluster
             "distinct_cluster_labels" - list of distinct cluster labels '''
def main_function(num_clusters, retokenize, recluster, corpusdir, dataset_path, n_words, minibatch, num_processes):
    try:
        nltk.data.find('tokenizers/stopwords')
    except:
        stopwords = nltk.download('stopwords')
    
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt')

    stemmer = SnowballStemmer("english")

    dataset_name, file_place = initialize_output_location(dataset_path)
    trailer_text = dataset_name + "_k=" + str(num_clusters)
    
    print("\nAll outputs generated will be in \"~\\cluster-datalake-outputs\\" + dataset_name + "--output\"")

    #=========1=========2=========3=========4=========5=========6=======

    # tokenize and cluster    
    fnames, dataset = to_retokenize(retokenize, corpusdir, dataset_path, num_processes)
    tfidf_matrix = np.load(os.path.join(file_place, "tfidf_matrix_" + dataset_name + ".npy")).item()
    to_recluster(num_clusters, retokenize, recluster, tfidf_matrix, dataset_path, minibatch)
    
    # load in existing saved files
    km = joblib.load(os.path.join(file_place, 'doc_cluster_' + trailer_text + '.pkl'))
    vocab_frame = pd.read_pickle(os.path.join(file_place, "vocab_frame_" + dataset_name + ".pkl"))
    terms = np.load(os.path.join(file_place, "terms_" + dataset_name + ".npy")).tolist()
    dist = np.load(os.path.join(file_place, "distance_matrix_" + dataset_name + ".npy"))
    print("\nLoaded in existing dependencies...\n")

    clusters = km.labels_.tolist()

    # get the actual number of clusters in the dataframe
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

    #=========1=========2=========3=========4=========5=========6=======
   
    # open file writer for result output
    fwriter = open(os.path.join(file_place, "doc_clusters_" + trailer_text + ".txt"), "w")
    fwriter.write("Clusters from text files in: " + corpusdir)

    fwriter.write("\nTop terms per cluster: \n\n")
    print("Top terms per cluster: \n")

    #sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
   
    all_cluster_words = {}
    # for each cluster

    ''' terms contains all the feature labels of the clustering
        vocab_frame contains all the tokens mapped to their stemmed counterparts
        you're finding the token version of the stem you get
        the stem you get is at the ind position of the terms list
        ind is from order_centroids
        order_centroids is a sorted array with num_clusters rows and len(terms) features
        order_centroids[i] is the coordinates of cluster i
        order_centroids[i,:] is the coordinates from ALL features for cluster i'''
         
    distinct_cluster_labels = sorted(distinct_cluster_labels)
    for i in distinct_cluster_labels:
        fwriter.write("Cluster " + str(i) + " words: ")
        print("Cluster %d words:" % i, end='')
        print("")
        print("=======================")
        print("DEBUGGING. ")
        
        print("length of terms: ", len(terms))
        print("lengh of index of vocab_frame: ", len(vocab_frame.index))

        
        for ind in order_centroids[i, : n_words]:
            #test_var = vocab_frame.ix[terms[ind].split(" ")].values.tolist()[0]
            
            print(' %s' % terms[ind].split(' '))

        print("=======================")
        
        cluster_words = [] 
        seen = []
        # print the first "n_words" words in a cluster
        for ind in order_centroids[i, : n_words]:
            #test_var = vocab_frame.ix[terms[ind].split(" ")].values.tolist()[0]
            
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=",")
            fwriter.write(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].rstrip('\n') + ", ")
            cluster_words.append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0])
        print()
        fwriter.write("\n")
         
        all_cluster_words.update({i:cluster_words})

        # print out the filenames in the cluster
        print("Cluster %d filenames:" % i, end='')
        fwriter.write("Cluster " + str(i) + " filenames: ")
        for filename in frame.loc[i]['filename'].values.tolist():
            print(' %s,' % filename, end='')
            fwriter.write(filename.rstrip('\n') + ", ")
        print("\n")
        fwriter.write("\n\n") 

    fwriter.close()
    print("Output written to \"doc_clusters_" + trailer_text + ".txt\"")

    #=========1=========2=========3=========4=========5=========6========
    
    if retokenize == "1":
        print("\nBypassing MDS fitting...") 
        # multidimensional scaling: convert distance matrix into 3-dimensions
        mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
        print("\nFitting the distance matrix into 3 dimensions...")
        pos_save = mds.fit_transform(dist)  # shape (n_components, n_samples)
        np.save(os.path.join(file_place, "mds_pos_" + trailer_text + ".npy"), pos_save)

    print("Loaded existing MDS fit.")
    pos = np.load(os.path.join(file_place, "mds_pos_" + trailer_text + ".npy")).item()
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    # set up plot
    fig = plt.figure(figsize=(17,9))
    ax = Axes3D(fig)

    # create data frame with MDS results, cluster numbers, and filenames
    df = pd.DataFrame(dict(x=xs, y=ys, z=zs, label=clusters, filename=fnames)) 
    # group by cluster
    groups = df.groupby('label')

    # for each cluster, plot the files in that cluster
    print("\n\nPlotting scatterplot of cluster points...")
    for name, group in tqdm(groups):
        # color = ('#%06X' % random.randint(0,256**3-1))
        color = np.random.rand(3,)
        for t in range(group.shape[0]):
            ax.scatter(group.x.iloc[t], group.y.iloc[t], group.z.iloc[t], 
                c=color, marker='o')
            ax.set_aspect('auto')

    plt.savefig(os.path.join(file_place, "3D_document_cluster_" + trailer_text + ".png"), dpi=300)
    print("Scatter plot written to \"3D_document_cluster_" + trailer_text + ".png\"")
           
    return frame, all_cluster_words, distinct_cluster_labels

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETERS: "frame" - a dataframe containing which files are in which clusters
                "num_clusters" - the number of clusters
                "home_dir" - the parent directory of the dataset (e.g. /home/ljung/)
                "dataset_path" - the directory of the dataset (e.g. /home/ljung/pub8/)
    DOES: plots a pdf with all the bar charts on it
          each bar chart shows which directories the files in a cluster come from '''
def bar_clusters(frame, distinct_cluster_labels, num_clusters, home_dir, dataset_path):
    plt.figure("bar")
  
    matplotlib.rcParams.update({'font.size': 4})
    
    dataset_name, file_place = initialize_output_location(dataset_path)
    trailer_text = dataset_name + "_k=" + str(num_clusters)
    
    print("\n\nGenerating barcharts...")    
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(file_place, "text_barcharts_" + trailer_text + ".pdf"))
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
                path = path_utilities.remove_path_end(row['filename'])
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
    np.save(os.path.join(file_place, "cluster_directories_" + trailer_text + ".npy"), cluster_directories)
    print("Bar charts written to \"text_barcharts_" + trailer_text + ".pdf\"")

#=========1=========2=========3=========4=========5=========6=========7=

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

#=========1=========2=========3=========4=========5=========6=========7=

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

def print_cluster_stats(frame, top_words, dataset_path, num_clusters):
    dataset_name, file_place = initialize_output_location(dataset_path)
    trailer_text = dataset_name + "_k=" + str(num_clusters)
    cluster_directories = np.load(os.path.join(file_place, "cluster_directories_" + trailer_text + ".npy"))
    
    fr = open(os.path.join(file_place, "cluster_stats_" + trailer_text + ".txt"), "w")
    total_silhouette, scores = silhouette.compute_silhouette(cluster_directories, dataset_path)
    frqscores, total_frq_score = frequencydrop.compute_freqdrop_score(cluster_directories)
    num_files_per_cluster = frame['cluster'].value_counts().sort_index().tolist()

    print("\n\nComputing cluster statistics...")
    for clust_num in tqdm(range(len(cluster_directories))):
        c_stats = "Cluster " + str(clust_num) + "\n"
        c_stats = c_stats + str(num_files_per_cluster[clust_num]) + " files \n"
        c_stats = c_stats + get_cluster_stats(cluster_directories[clust_num])
        c_stats = c_stats + "\nSilhouette score: " + str(scores[clust_num])
        c_stats = c_stats + "\nFrequency drop score: " + str(frqscores[clust_num])
        c_stats = c_stats + "\nTop 10 words: " + ", ".join(top_words.get(clust_num))
        fr.write(c_stats + "\n\n")
    ensemble_score = ((total_silhouette+1)/2 + total_frq_score)/2
    fr.write("\nTotal silhouette score: " + str(total_silhouette))
    fr.write("Average frequency drop score: " + str(total_frq_score))
    fr.write("Total ensemble score: " + str(ensemble_score))
    fr.close()
    print("Cluster stats written to \"cluster_stats_" + trailer_text + ".txt\"")
    return ensemble_score

#=========1=========2=========3=========4=========5=========6=========7=

# MAIN PROGRAM

def runflow(num_clusters, retokenize, recluster, dataset_path, minibatch, num_processes):
    
    if retokenize.lower() == 'y':
        retokenize = "1"
    else:
        retokenize = "0"

    if recluster.lower() == 'y':
        recluster = "1"
    else:
        recluster = "0"

    if minibatch.lower() == 'y':
        minibatch = "1"
 
    home_dir = path_utilities.remove_path_end(dataset_path)
    corpusdir = os.path.join(home_dir, "converted-" +  path_utilities.get_last_dir_from_path(dataset_path)) #eg. /home/ljung/converted/
 
    # record initial time that program started
    t0 = time()
    
    fr, all_cluster_words, distinct_cluster_labels = main_function(num_clusters, retokenize, recluster, corpusdir, dataset_path, 10, minibatch, num_processes)
    bar_clusters(fr, distinct_cluster_labels, num_clusters, home_dir, dataset_path)    
    ensemble_score = print_cluster_stats(fr, all_cluster_words, dataset_path, num_clusters)
    
    # print total time taken to run program
    print("\nTime taken: ", time()-t0, " seconds\n")
    return ensemble_score

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    num_clusters = int(sys.argv[1])
    retokenize = sys.argv[2]
    recluster = sys.argv[3]
    # the directory containing the files not in pub8 you want to cluster
    dataset_path = sys.argv[4] #eg. /home/ljung/pub8
    print("Don't run this file standalone.")
    #runflow(num_clusters, retokenize, recluster, dataset_path) 


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import random
import numpy as np
import pandas as pd
from time import time
from glob import glob
from six import string_types
import nltk, re, os, codecs, mpld3, sys
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from nltk.stem.snowball import SnowballStemmer
from mpl_toolkits.mplot3d import Axes3D

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAM: a string with a directory
    RETURNS: only the last folder name '''
def get_end_directory(directory):
    ind = 0
    ct = 0
    for ch in directory[::-1]:
        if ch=="/":
            ct+=1
            if ct>1:
                break
        ind += 1
    return directory[len(directory)-ind:len(directory)-1].replace("/","")

''' PARAM: a full path
    RETURNS: only the filename '''
def get_fname_from_path(path):    
    filename = ""
    for c in path[::-1]:
        if c=="/" or c=="@":
            break
        filename = c+filename
    return filename

''' PARAMETER: a single filename. 
    RETURNS: the file without its extension '''
def remove_extension(filename):    
    length = len(filename)
    for ch in filename[::-1]:
        if ch==".":
            break
        length = length - 1 
    return filename[:length-1]

def str_decode(string):
    return string.replace("@","/")

def get_immediate_subdirectories(a_dir):
    sub_list = []
    for name in os.listdir(a_dir):
        if os.path.isdir(os.path.join(a_dir, name)):
            sub_list.append(os.path.join(a_dir, name)+"/")
    return sub_list

''' PARAM: a parent directory containing /pdf/ /doc/ etc
    RETURNS: a list of filenames and a list of the contents of the files 
    DOES: gets all the filenames and their contents of a directory'''   
def get_document_contents(directory):
    ext_paths = np.load("/home/ljung/CDIAC-clust/paths_work/extension_index.npy").item()
    filenames = []
    data = []
    i = 1
    
    # a list of the paths of the txt files still in pub8 
    txt_paths = ext_paths.get("txt")
    for path in txt_paths:
        if os.path.isfile(path):
            i = i+1
            print("txt ", i) 
            # add the filename to "filenames" 
            #filenames.append(get_fname_from_path(path))
            filenames.append(path)
            # read the contents of the file and remove newlines
            freader = open(path, "r", errors='backslashreplace')
            contents = freader.read()#.encode("utf-8").decode('utf-8', 'backslashreplace')
            freader.close()
            contents = contents.replace("\n","")
            # add the string of the contents of the file to "data"
            data.append(contents)
    
    # for every file in the directory of converted files
    conv_folders = get_immediate_subdirectories(directory)
    print(directory, conv_folders)
    for folder in conv_folders:
        filetype = get_end_directory(folder)
        if filetype in ["pdf", "doc", "docx"]:
            for filename in os.listdir(folder):
                current_file = os.path.join(folder,filename)
                if os.path.isfile(current_file):
                    i = i + 1
                    print(filetype, i)
                    # add the non-converted filename to "filenames" 
                    new_name = str_decode(remove_extension(filename))#+"."+filetype
                    filenames.append(new_name)
                    # read the contents of the file and remove newlines
                    freader = open(current_file, "r", errors='backslashreplace')
                    contents = freader.read()#.encode("utf-8").decode('utf-8', 'backslashreplace')
                    freader.close()
                    contents = contents.replace("\n","")
                    # add the string of the contents of the file to "data"
                    data.append(contents)
    
    print("num total files: ", i)
    print("All directory contents retrieved")
    return filenames, data

''' PARAM: the text of a document
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
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

''' PARAM: the text of a document
    RETURN: a list of filtered tokens
    DOES: tokenizes the document only (doesn't stem) '''
def tokenize_only(text):
    # tokenize by sentence, then by word so punctuation is its own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out tokens without letters (e.g., numbers, punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#=========1=========2=========3=========4=========5=========6=========7=

def main_function(num_clusters, retokenize, corpusdir):
    # record initial time that program started
    t0 = time()

    # gets the filenames and their contents
    fnames, dataset = get_document_contents(corpusdir)
    #stopwords = nltk.download('stopwords')

    #nltk.download('punkt')
    stemmer = SnowballStemmer("english")

    #=========1=========2=========3=========4=========5=========6=======

    if retokenize == "1": 
        totalvocab_stemmed = []
        totalvocab_tokenized = []
        count = 1
        d_length = len(dataset)
        for i in dataset:
            print("tokenizing document ", count, " of ", d_length)
            count = count + 1
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
        vocab_frame.to_pickle("vocab_frame.pkl")
        print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

        #define vectorizer parameters
        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                              min_df=0.2, stop_words='english', use_idf=True, 
                              tokenizer=tokenize_and_stem, ngram_range=(1,3))

        #fits the vectorizer to the dataset
        tfidf_matrix = tfidf_vectorizer.fit_transform(dataset) 
        terms = tfidf_vectorizer.get_feature_names()
        np.save("terms.npy", terms)
        dist = 1 - cosine_similarity(tfidf_matrix)
        np.save("distance_matrix.npy", dist)
        print("vectorizer fitted to data")

        #=========1=========2=========3=========4=========5=========6======

        # cluster using KMeans on the tfidf matrix
        km = KMeans(n_clusters=num_clusters)
        km.fit(tfidf_matrix)
        clusters = km.labels_.tolist()
        print("kmeans clustering complete")

        # pickle the model, reload the model/reassign the labels as the clusters
        joblib.dump(km,  'doc_cluster.pkl')


    km = joblib.load('doc_cluster.pkl')
    vocab_frame = pd.read_pickle("vocab_frame.pkl")
    terms = np.load("terms.npy")
    dist = np.load("distance_matrix.npy")
    print("Loaded in existing cluster profile...\n")

    clusters = km.labels_.tolist()

    # create a dictionary "db" of filenames, contents, and clusters
    db = {'filename': fnames, 'content': dataset, 'cluster': clusters}
    # convert "db" to a pandas datafram
    frame = pd.DataFrame(db, index=[clusters], columns=['filename','cluster'])
    # print the number of files in each cluster
    print(frame['cluster'].value_counts())

    #=========1=========2=========3=========4=========5=========6=========
    ''' 
    # open file writer for result output
    # output_path = os.path.join(corpusdir, "results/")
    output_path = "results/"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    os.chdir(output_path)
    fwriter = open("doc_clusters.txt", "w")
    fwriter.write("clusters from text files in: " + corpusdir)

    fwriter.write("\nTop terms per cluster: \n\n")
    print("Top terms per cluster: \n")

    #sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

    # for each cluster
    for i in range(num_clusters):
        fwriter.write("Cluster " + str(i) + " words: ")
        print("Cluster %d words:" % i, end='')
        
        # print the first "n_words" words in a cluster
        n_words = 10
        for ind in order_centroids[i, : n_words]:
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0],
                    end=",")
            fwriter.write(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].rstrip('\n') + ", ")
        print()
        fwriter.write("\n")
        
        # print out the filenames in the cluster
        print("Cluster %d filenames:" % i, end='')
        fwriter.write("Cluster " + str(i) + " filenames: ")
        for filename in frame.ix[i]['filename'].values.tolist():
            print(' %s,' % filename, end='')
            fwriter.write(filename.rstrip('\n') + ", ")
        print("\n")
        fwriter.write("\n\n")

    fwriter.close()
    print("output written to \"doc_clusters.txt\" in \"results\"")

    #=========1=========2=========3=========4=========5=========6========

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
    for name, group in groups:
        # color = ('#%06X' % random.randint(0,256**3-1))
        color = np.random.rand(3,)
        for t in range(group.shape[0]):
            ax.scatter(group.x.iloc[t], group.y.iloc[t], group.z.iloc[t], 
                c=color, marker='o')
            ax.set_aspect('auto')

    plt.savefig("3D_document_cluster", dpi=300)
    print("scatter plot written to \"3D_document_cluster.png\"")
    
    # print total time taken to run program
    print("time taken: ", time()-t0)
    '''
    return frame

def bar_clusters(frame, path, num_clusters):
    #file_pathtokens_dict, file_path_dict = DFS(path,1)
    #file_paths = DFS(path)
    plt.figure("bar")
  
    output_path = "txt_cluster_bars/"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    os.chdir(output_path)
    matplotlib.rcParams.update({'font.size': 6})
    
    # for each cluster, generate a bar chart 
    for i in range(num_clusters):
        plt.clf()
        paths_in_cluster = {}
        # get the files associated with the current cluster
        cluster_files = frame.loc[frame['cluster']==i]
        count = 0 
        for index, row in cluster_files.iterrows():
            if count>1:
                path = get_dir_from_path(row['filename'])
                # print("path: ", path)
                # if the path is already in the cluster, add to count
                if path in paths_in_cluster:
                    paths_in_cluster.update({path:paths_in_cluster.get(path)+1})
                else:
                    paths_in_cluster.update({path:1})
            count+=1
        sorted_names = []
        sorted_counts = []
        # sort the paths in ascending order based on # of occurrences
        for e in sorted(paths_in_cluster, key=paths_in_cluster.get, reverse=True):
            sorted_names.append(e)
            sorted_counts.append(paths_in_cluster[e])

        print(sorted_names, sorted_counts)
        y_pos = np.arange(len(sorted_names))

        plt.bar(y_pos, sorted_counts, align='center', alpha=0.5)
        plt.xticks(y_pos, sorted_names, rotation=90)
        plt.rc('xtick', labelsize=3)
        plt.ylabel('Number of files')
        plt.title('Directories in Cluster ' + str(i))
        plt.tight_layout()
        save_name = "barchart_cluster"+str(i)       
        # plt.savefig(save_name, dpi=200)
        
        pdf = matplotlib.backends.backend_pdf.PdfPages("text_barcharts.pdf")
        for fig in range(1, plt.gcf().number+1): ## will open an empty extra figure :(
            pdf.savefig(fig)
        pdf.close()

''' PARAM: a full path
    RETURNS: the path without the filename '''
def get_dir_from_path(path):   
    ind = 0
    for c in path[::-1]:
        ind = ind+1
        if c=="/" or c=="@":
            break
    return path[:len(path)-ind+1]


# MAIN PROGRAM

def main():
    num_clusters = int(sys.argv[1])
    retokenize = sys.argv[2]
    # the directory containing the files not in pub8 you want to cluster
    corpusdir = sys.argv[3] #eg. /home/ljung/extension_sorted_data/
    fr = main_function(num_clusters, retokenize, corpusdir)
    bar_clusters(fr, "/home/ljung/pub8/", num_clusters)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 


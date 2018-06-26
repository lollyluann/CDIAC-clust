import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
import nltk, re, os, codecs, mpld3, sys
from time import time
from six import string_types
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from nltk.stem.snowball import SnowballStemmer
from mpl_toolkits.mplot3d import Axes3D

#=========1=========2=========3=========4=========5=========6=========7=

num_clusters = int(sys.argv[1])
# the directory of the files you want to cluster
corpusdir = "/home/ljung/extension_sorted_data/test_pdfs/"


''' PARAM: a string containing the directory of .txt files
    RETURNS: a list of filenames and a list of the contents of the files 
    DOES: gets all the filenames and their contents of a directory'''   
def get_document_contents(directory):
    filenames = []
    data = []
    # for every file in the given directory
    print("num files in directory: ", len(os.listdir(directory)))
    i = 1
    for filename in os.listdir(directory):
        print(i)
        i = i + 1
        current_file = os.path.join(directory,filename)
        if os.path.isfile(current_file):
            # add the filename to "filenames" 
            filenames.append(filename)
            # read the contents of the file and remove newlines
            freader = open(current_file, "r")
            contents = freader.read()
            freader.close()
            contents = contents.replace("\n","")
            # add the string of the contents of the file to "data"
            data.append(contents)
    print("Directory contents retrieved")
    return filenames, data

''' PARAM: the text of a document
    RETURN: list of stems
    DOES: splits a document into a list of tokens & stems each token '''
def tokenize_and_stem(text):
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

# record initial time that program started
t0 = time()

# gets the filenames and their contents
fnames, dataset = get_document_contents(corpusdir)
#stopwords = nltk.download('stopwords')

#nltk.download('punkt')
#stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

#=========1=========2=========3=========4=========5=========6=========7=

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
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                      min_df=0.2, stop_words='english', use_idf=True, 
                      tokenizer=tokenize_and_stem, ngram_range=(1,3))

#fits the vectorizer to the dataset
tfidf_matrix = tfidf_vectorizer.fit_transform(dataset) 
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)
print("vectorizer fitted to data")

#=========1=========2=========3=========4=========5=========6=========7=

# cluster using KMeans on the tfidf matrix
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print("kmeans clustering complete")

# pickle the model, reload the model/reassign the labels as the clusters
# uncomment the line below to resave model
joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

# create a dictionary "db" of filenames, contents, and clusters
db = {'filename': fnames, 'content': dataset, 'cluster': clusters}
# convert "db" to a pandas datafram
frame = pd.DataFrame(db, index=[clusters], columns=['filename','cluster'])
# print the number of files in each cluster
print(frame['cluster'].value_counts())

#=========1=========2=========3=========4=========5=========6=========7=

# open file writer for result output
os.chdir("/home/ljung/CDIAC-clust/")
fwriter = open("pdf_clusters.txt", "w")
fwriter.write("clusters from pdf files in: " + corpusdir)

fwriter.write("\nTop terms per cluster: \n\n")
print("Top terms per cluster: \n")

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

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
print("output written to \"pdf_clusters.txt\"")



mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

'''cluster_colors = {}
for r in range(num_clusters):
    color = ('#%06X' % random.randint(0,256**3-1))
    cluster_colors.update({r:color})
    print(color)
'''

fig = plt.figure(figsize=(20,10))
ax = Axes3D(fig)
#ax.scatter(xs, ys, zs, s=fig.dpi/72., marker='.')

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, z=zs, label=clusters, filename=fnames)) 
#group by cluster
groups = df.groupby('label')
# set up plot
#fig, ax = plt.subplots(figsize=(17, 9)) # set size

#iterate through groups to layer the plot
for name, group in groups:
    print("name: ", name, " group: ", group)
    color = ('#%06X' % random.randint(0,256**3-1))
    for piece in group:
        ax.scatter(group.x, group.y, group.z, marker='.', color=color)
    #ax.set_aspect('auto')

plt.savefig("pdf_3D_cluster", dpi=1000)






# print total time taken to run program
print("time taken: ", time()-t0)

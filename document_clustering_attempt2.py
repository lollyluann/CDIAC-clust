import numpy as np
import pandas as pd
import nltk, re, os, codecs, mpld3, sys
from os.path import isfile, join
from six import string_types
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer

#=========1=========2=========3=========4=========5=========6=========7=

corpusdir = "/home/ljung/extension_sorted_data/test_pdfs/"

def get_document_contents(directory):
    filenames = []
    data = []
    for filename in os.listdir(directory):
        current_file = join(directory,filename)
        if isfile(current_file):
            filenames.append(filename)
            freader = open(current_file, "r")
            contents = freader.read()
            freader.close()
            contents = contents.replace("\n","")
            #contents.replace(" ","")
            data.append(contents)
    return filenames, data

fnames, dataset = get_document_contents(corpusdir)
#stopwords = nltk.download('stopwords')
#nltk.download('punkt')
#stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in dataset:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(dataset) #fit the vectorizer to synopses
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)


num_clusters = int(sys.argv[1])
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

db = { 'filename': fnames, 'content': dataset, 'cluster': clusters}
frame = pd.DataFrame(db, index = [clusters] , columns = ['filename', 'cluster'])
print(frame['cluster'].value_counts())


print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :10]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=",") #.encode('utf-8', 'ignore'), end=',')
    print()
    
    print("Cluster %d filenames:" % i, end='')
    for filename in frame.ix[i]['filename'].values.tolist():
        print(' %s,' % filename, end='')
    print()

    print() 
print()
print()

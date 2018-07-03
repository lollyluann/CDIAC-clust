import generate_shortened_token_dict
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pylab as pl
import numpy as np
import sys
import os

''' PARAMETER: a single filename. 
    RETURNS: the file extension from that filename. '''
def get_single_extension(filename):    
    extension = ""
    length = len(filename)
    for ch in filename[::-1]:
        if ch==".":
            break
        # "extension" contains just the extension from a filename
        extension = ch + extension
        length = length - 1 
    return extension.lower()

''' PARAMETER: a single filename. 
    RETURNS: the file without its extension '''
def remove_extension(filename):    
    length = len(filename)
    for ch in filename[::-1]:
        if ch==".":
            break
        length = length - 1 
    return filename[:length-1]

''' PARAMETER: a list of all filenames with extensions
    RETURNS: a list of filenames without the extensions '''
def remove_file_extension(filenames):
    filenames_no_extensions = []
    extensions = []
    for filename in filenames:
        extension = ""
        length = len(filename)
        for ch in filename[::-1]:
            if ch==".":
                break
            # "extension" contains just the extension from a filename
            extension = ch + extension
            length = length - 1 
        filenames_no_extensions.append(filename[0:length-1])
        extensions.append(extension.lower())
        #print(filename)
        #print(extension)
    
    return filenames_no_extensions, extensions

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a single filename
    RETURNS: a list of all tokens with "_" "-" and "._" removed '''
def generate_tokens(filename):
    # replace dashes with underscores, and .dashes with underscores
    # in each filename
    filename = filename.replace("-","_").replace("._","_")
    # create a list of tokens, splitting by underscores
    old_tokens =  filename.split("_")
    new_tokens = []
    # removes all empty tokens
    for x in old_tokens:
        if x!="":
            new_tokens.append(x)
    return new_tokens
        
#=========1=========2=========3=========4=========5=========6=========7=


''' PARAMETER: a list of filenames
    DOES: sorts a dictionary with the counts of each token
    RETURNS: a list of sorted tokens and a list of sorted counts '''
def count_and_sort_tokens(filenames):
    # a dict mapping tokens to the count of how many times they appear
    token_count_dict = {}
    # for each filename
    for fn in filenames:
        #split the filename into tokens using delimiters like "_"
        tokens = set(generate_tokens(fn))
        # for each token
        for token in tokens:
            # if the token is already in our dict
            if token in token_count_dict.keys():
                # grab the old count
                old_count = token_count_dict.get(token)
                # increment and update the count in the dict              
                token_count_dict.update({token:old_count+1})
            else:
                # otherwise, add a key,value pair with a count of 1
                token_count_dict.update({token:1})
    sorted_tokens = []
    sorted_counts = []
    # for each token in the dict, iterating from largest to smallest count 
    for w in sorted(token_count_dict, key=token_count_dict.get, reverse=False):
        # add the token to a sorted list of tokens
        sorted_tokens.append(w)
        # add the corresponding count to a list of counts
        sorted_counts.append(token_count_dict[w])    
        print(w, token_count_dict[w])

    # log-scaled bins
    bins = np.logspace(0, 4, 100)
    widths = (bins[1:] - bins[:-1])
    #print(bins)
    #print(widths)

    # Calculate histogram
    hist = np.histogram(sorted_counts, bins=bins)
    # normalize by bin width
    hist_norm = hist[0]/widths

    # plot it!
    plt.figure("hist")
    plt.clf()
    plt.bar(bins[:-1], hist[0], widths)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Token Document Frequency")
    plt.xlabel("# of files a word appears in")
    plt.ylabel("Word count")

    plt.savefig("hist-o-gram",dpi=500)
   
    return sorted_tokens, sorted_counts


''' PARAMETER: a list of extensions
    DOES: sorts a dictionary with the counts of each token
    RETURNS: a list of sorted tokens and a list of sorted counts '''
def count_and_sort_exts(filenames):
    # a dict mapping tokens to the count of how many times they appear
    token_count_dict = {}
    # for each filename
    for fn in filenames:             
        # if the token is already in our dict
        if fn in token_count_dict.keys():
            # grab the old count
            old_count = token_count_dict.get(fn)
            # increment and update the count in the dict              
            token_count_dict.update({fn:old_count+1})
        else:
            # otherwise, add a key,value pair with a count of 1
            token_count_dict.update({fn:1})
    sorted_tokens = []
    sorted_counts = []
    # for each token in the dict, iterating from largest to smallest count 
    for w in sorted(token_count_dict, key=token_count_dict.get, reverse=False):
        # add the token to a sorted list of tokens
        sorted_tokens.append(w)
        # add the corresponding count to a list of counts
        sorted_counts.append(token_count_dict[w])    
        print(w, token_count_dict[w])

    return sorted_tokens, sorted_counts

#=========1=========2=========3=========4=========5=========6=========7=

def plot_extension_pie(extensions, num_slices):
    sorted_exts, sorted_counts = count_and_sort_exts(extensions)
    labels = []
    sizes = []
    for x in range(num_slices):
        labels.append(sorted_exts[x])
        sizes.append(sorted_counts[x])
    plt.figure("pie")
    plt.clf()
    plt.pie(sizes,labels=labels)
    plt.axis('equal')
    plt.title(str(num_slices) + " Most Common Extensions in CDIAC")
    plt.savefig("pie",dpi=300)

    f = open('top_exts.txt','w')
    for i in range(num_slices):
        f.write(sorted_exts[i] + "\n")
    f.close()

#=========1=========2=========3=========4=========5=========6=========7=

# MAIN FUNCTION
def main():
    root_path = sys.argv[1]
    token_length = int(sys.argv[2])
    allpaths = generate_shortened_token_dict.DFS(root_path, token_length)

    # a list of all the file names (without the paths)
    filenames = list(allpaths[1].keys())
    filenames_no_ext, exts = remove_file_extension(filenames)


    os.chdir("/home/ljung/CDIAC-clust/")
    f = open("filenames.txt", "w")
    for x in filenames_no_ext:
        f.write(x+ "\n")
        #print(generate_tokens(x))
    f.close()

    sorted_tokens, sorted_counts = count_and_sort_tokens(filenames_no_ext)

    token_file = open("variable_length_tokens.txt", "w")
    for token in sorted_tokens:
        token_file.write(token + "\n")
    token_file.close()

    plot_extension_pie(exts,15)


if __name__ == "__main__":
    main()

import numpy as np
import tokenizer
import sys
import os

# FIRST ARGUMENT: the root directory path.

#=========1=========2=========3=========4=========5=========6=========7=

''' RETURNS: a list of the paths of every file in the directory "path". '''
def DFS(path):
    stack = []
    all_paths = []
    stack.append(path);
    while len(stack) > 0:
        # pop a path off the stack 
        tmp = stack.pop(len(stack) - 1)
        # if this is a valid path.
        if(os.path.isdir(tmp)):
            # for every item in the "tmp" directory
            for item in os.listdir(tmp):
                # throws the path given by "tmp" + "item" onto the stack
                # for each "item".
                stack.append(os.path.join(tmp, item))
        # if it's not a valid path but it IS a valid file.
        elif(os.path.isfile(tmp)):
            # "tmp" format: '/home/ljung/shit/shitposting/'
            all_paths.append(tmp)
    return all_paths

''' RETURNS: a dictionary which maps extension names of the form "csv"
             to lists of the full paths of files with those extensions in 
             the dataset.'''
def extension_indexer(root, n):  
    allpaths = DFS(root)
    
    # a list of all the filenames (without paths)
    filenames = []
    for path in allpaths:
        filenames.append(get_fname_from_path(path))
    filenames_no_ext, exts = tokenizer.remove_all_extensions(filenames)
    
    sorted_exts, sorted_counts = tokenizer.count_and_sort_exts(exts, n)
    top_n_exts = sorted_exts[:n]
     
    # makes a dictionary key for each of the top extensions
    ext_locations = {}
    for extension in top_n_exts:    
        ext_locations.update({extension:[]})     

    # checks every file and saves the paths of those with the top extensions
    # in a dict called "ext_locations"
    for fp in allpaths:
        fn = get_fname_from_path(fp)
        if fn[:2]!="._":
            ext = tokenizer.get_single_extension(fn)
            if ext in top_n_exts:
                ext_list = ext_locations.get(ext)
                ext_list.append(fp)
                ext_locations.update({ext:ext_list})
   
    np.save("extension_index.npy", ext_locations)
    
    return ext_locations

#=========1=========2=========3=========4=========5=========6=========7=

def str_encode(string):
    return string.replace("/","@")

def str_decode(string):
    return string.replace("@","/")
    
def get_fname_from_path(path):    
    filename = ""
    for c in path[::-1]:
        if c=="/" or c=="@":
            break
        filename = c+filename
    return filename

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    root_path = sys.argv[1]
    num_slices = sys.argv[2]

    # allpaths = DFS(root_path)
    
    '''os.chdir(root_path)
    f1 = open(root_path + "transformed_paths.txt","w")

    f1.write(str(allpaths))
    f1.close()'''

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 

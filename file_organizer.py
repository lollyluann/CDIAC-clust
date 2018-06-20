from shutil import copyfile
import generate_token_dict
import tokenizer
import sys
import os

#=========1=========2=========3=========4=========5=========6=========7=

path = sys.argv[1]

# gets a dictionary of the files to their paths
file_pathtokens_dict, file_path_dict = generate_token_dict.DFS(path,1)
#paths_to_move = []

# reads in the top file extensions
ext_file = open("top_exts.txt", "r")
lines  = ext_file.readlines()
top_n_exts = []
# removes newline characters
for e in lines:
    top_n_exts.append(e.rstrip())

# makes a new directory for the sorted data
path = "/home/ljung/extension_sorted_data/"
if not os.path.isdir(path):
    os.mkdir(path)
# makes a new folder for each of the top extensions
for p in top_n_exts:
    if not os.path.isdir(path+p+"/"):
        os.mkdir(path+p+"/")

# checks every file and moves those with the top extensions
    # e.g. /home/bob.pdf moves to /home/files/pdf/bob.pdf
for fn in file_path_dict.keys():
    ext = tokenizer.get_single_extension(fn)
    if ext in top_n_exts:
        #paths_to_move.append(file_path_dict.get(fn))
        if not os.path.exists(path+ext+"/"+fn):
            copyfile(file_path_dict.get(fn),path+ext+"/"+fn)


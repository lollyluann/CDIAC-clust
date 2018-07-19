from tqdm import tqdm
import random
import csv
import sys
import os

#=========1=========2=========3=========4=========5=========6=========7=

# Generates a new dataset in "dest" called "n-ary_toy_dataset" which
# is a directory hierarchy with n-ary tree structure. The leaf nodes 
# contain 100 text files each, where each text file contains 100 words
# per line and 100 lines. Each word in every text file in a given leaf
# node is the same, taken from a list of seed words.

# A .csv file called "seed_words.csv" should be in the same directory
# as this script for proper functionality.  

#=========1=========2=========3=========4=========5=========6=========7=

# ARGUMENTS
# the directory in which to place the toy dataset
dest = sys.argv[1]
try:
    n = int(sys.argv[2])
    depth = int(sys.argv[3])
except TypeError:
    print("You probably tried to pass a non-integer argument. These"
          + " are supposed to be natural numbers. ")
    exit()

# DOES: checks whether or not a directory argument is valid 
def check_valid_dir(some_dir):
    if not os.path.isdir(some_dir):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST EIN UNGÜLTIGES VERZEICHNIS!!!!")
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

# check if destination is valid, get its absolute path
check_valid_dir(dest)
dest = os.path.abspath(dest)
# generate path to the new root of our toy dataset
root_name = str(n) + "-ary_toy_dataset"
root_path = os.path.join(dest, root_name)
# make sure a directory with the same name doesn't already exist
if os.path.isdir(root_path):
    print("This directory already exists, change the existing "
          + "directory's name, or try a different destination. ")
    exit()
# create the directory
os.system("mkdir " + root_path)

#=========1=========2=========3=========4=========5=========6=========7=

def read_seed():
    word_list = []
    with open('seed_words2.csv', 'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in rows: 
            for word in row:
                word_list.append(word)
    return word_list

#=========1=========2=========3=========4=========5=========6=========7=

def generate_dataset(root_path, num_children, depth, word_list):
    if (depth == 1):
        word = random.choice(word_list)
        row_list = []
        for k in tqdm(range(10)):
            row_list.append(word)
        word_string = ' '.join(token for token in row_list) + " "
        text_file_path = os.path.join(root_path, word + "_1" + ".txt")
        with open(text_file_path, 'a') as the_file: 
            for j in tqdm(range(100)):
                the_file.write(word_string + "\n")
        for i in tqdm(range(20)):
            if i != 1:
                next_file_path = os.path.join(root_path, 
                                              word + "_" 
                                              + str(i) + ".txt")
                os.system("cp " + text_file_path + " " + next_file_path)
    else:
        for i in tqdm(range(num_children)):
            child_path = os.path.join(root_path, str(i))
            os.system("mkdir " + child_path)
            generate_dataset(child_path, 
                             num_children, depth - 1, word_list)            
            
#=========1=========2=========3=========4=========5=========6=========7=

def main():
    print("Just the main function. ")
    word_list = read_seed()
    generate_dataset(root_path, n, depth, word_list)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 




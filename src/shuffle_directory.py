from path_utilities import remove_path_end

import numpy as np

import random
import math
import DFS
import sys
import os

# DOESN'T LIKE FILES WITH SPACES IN THE FILENAME, will skip. 

#=========1=========2=========3=========4=========5=========6=========7=

def parse_args():

    # arguments
    dataset_path = sys.argv[1]
    shuffle_ratio = float(sys.argv[2])
    print("Parsing arguments. ")
    
    arg_list = [
                dataset_path, 
                shuffle_ratio, 
                True, 
               ]
    print("Arguments parsed. ")
    return arg_list

#=========1=========2=========3=========4=========5=========6=========7=

#FUNCTIONS
# DOES: randomly shuffles the location of the files in 
#       "dataset_path". 
def shuffle(dataset_path, shuffle_ratio, warning, old_shuffle_tracker, filepaths):
    if warning == True:
        if (confirm(prompt="Warning, this will scramble the directory "
                    + "structure of all files and folders in " + dataset_path
                    + ". Are you sure you want to do this? ")):
            print("Ok.")
        else: exit()    
        if (confirm(prompt="Really sure, though?")):
            print("Ok.")
        else: exit()
        if (confirm(prompt="Super duper sure???")):
            print("Ok.")
        else: exit()



    # get a list of the paths to every file in the dataset
    # rooted at "dataset_path"
    new_filepaths = DFS.DFS(dataset_path)

    num_files = len(new_filepaths)
    print("Number of files: ", num_files)

    while len(old_shuffle_tracker) < len(filepaths):
        old_shuffle_tracker.append(0)  
    
    # we randomly shuffle the list of filepaths
    num_to_shuffle = math.floor(num_files * shuffle_ratio)
    print(num_to_shuffle)

    # only shuffle part of the dataset
    paths_to_shuffle = new_filepaths[0:num_to_shuffle] 
    
    # generate a permutation of the number of files
    perm = np.random.permutation(num_to_shuffle)
    perm2 = np.random.permutation(num_to_shuffle)

    # "num_to_shuffle" randomly chosen parent directories
    directory_list = []
   
    # for each file
    for i in range(num_to_shuffle):
        
        # get the image of the index under our permutation
        permuted_index = perm[i]
        
        # get its parent directory
        directory = remove_path_end(new_filepaths[permuted_index])
        
        # and add it to our list of parent directories
        directory_list.append(directory)
    
    # moves a random file somewhere in "directory_list" 
    for i in range(num_to_shuffle):
        
        # get the image of the index under our permutation
        permuted_index2 = perm2[i]
        
        # get the file we're moving
        next_file = "iiiiiiiiiiiiiiiiii"
        files_checked = 0
        while old_shuffle_tracker[permuted_index2] == 1:
            next_file = new_filepaths[permuted_index2]
            files_checked += 1
            if files_checked > 2000:
                break
        
        # get the randomly chosen destination directory
        dest_dir = directory_list[i]
        
        # move the file, only if dest dir isn't parent of next_file
        if remove_path_end(next_file) != dest_dir:
            os.system("mv \"" + next_file + "\" \"" + dest_dir + "\"")

    # create shuffle tracker 
    shuffled_DFS = DFS.DFS(dataset_path)
    shuffle_tracker = []
    for i in range(min([len(shuffled_DFS), len(filepaths)])):
        if shuffled_DFS[i] != filepaths[i]:
            shuffle_tracker.append(1)
        else:
            shuffle_tracker.append(old_shuffle_tracker[i])
    return shuffle_tracker

#=========1=========2=========3=========4=========5=========6=========7=

def confirm(prompt=None, resp=False):
    """prompts for yes or no response from the user. Returns True for yes and
    False for no.

    'resp' should be set to the default value assumed by the caller when
    user simply types ENTER.

    >>> confirm(prompt='Create Directory?', resp=True)
    Create Directory? [y]|n: 
    True
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: 
    False
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: y
    True

    """
    
    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')
        
    while True:
        ans = input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print ('please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False

#=========1=========2=========3=========4=========5=========6=========7=

#MAIN FUNCTION
def main():
    print("Just the main function. ")
    arg_list = parse_args()
    dataset_path = arg_list[0]
    shuffle_ratio = arg_list[1]
    warning = arg_list[2]
    shuffle(dataset_path, shuffle_ratio, warning)
  
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 

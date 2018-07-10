# paths\_work

Information about files and folders in this directory:

## paths\_clustering
Directory containing utilities and files strictly used for clutering on information gleaned from the paths of the files in the dataset. 

## converting\_utilities.py
Arguments:
[1] = top-level directory of dataset. 
[2] = destination directory, within which the script will write to a directory called `csv/` among others. 
[3] = k such that we take the k most frequent file extensions in the dataset to create an dict that maps extensions in the form `csv` to lists of the filepaths with these file extensions. 

Converts tabular files to `.csv` format and text files to `.txt` format so that they can be clustered later on. 

## extension\_index.npy
The saved dictionary which maps file extensions (strings) in the form `csv` to lists of all the paths of all the files in the dataset which end in this file extension. Only conatins the top k file extensions, where k is a user-specified positive integer passed as an argument to `converting_utilities.py`. 

## hist-o-gram.png
The histogram of n-gram tokens generated from all the filnames in the dataset, the y-axis is number of tokens, and the x-axis is the token-frequency: the number of different filenames within which a unique token appears. So the rightmost token is the most common token in all the filenames. The leftmost tokens, the ones with a frequency of 1, are tokens that only appear in a single filename. Both the axes and the bin-width are logarithmic, so that the plot will fit in a nice sized-image and still have meaningful features. 

## new\_DFS.py
Is not meant to be run. Contains functions which other functions call. Contains utilities to run DFS on the dataset and return a list of paths to each file. Also creates the dict `extension_index.npy` described above. It contains utilities to convert to the path-in-filename format `|home|ljung|pub8|a_file.txt`, and from it. It also contains a utility to grab the filename from a path. 

## new\_file\_organizer.py
DEPRECATED. 
Arguments:
[1] = top-level directory of dataset.
[2] = directory or location of directory to be created where we will store k directories named after the top k most frequent file extensions. 

Copies files with the most frequent file extensions and sorts them into new directories in the chosen location. Uses path-in-filename format.

## pie-full.png
Pie chart containing slices for every extension in the dataset. 

## pie.png
Pie chart containing slices for the top k extensions in the datset. 

## tokenizer.py
Arguments:
[1] = top-level directory of dataset. 
[2] = directory in which to plot pie charts. 

Generates variable length tokens by delimiting on special characters like `_`. Contains utilites to get extension from a filename, remove extension from a filename, and to remove all extensions from a list of filenames. Plots pie charts of file extension composition of the dataset, and also contains utilities to count and sort extensions as well as delimited tokens generated from the filenames. Can also plot histogram of these variable-length tokens.

## top\_exts.txt
List of the top k file extensions, plaintext, one per line. 

 

 

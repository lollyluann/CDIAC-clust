import sys
import os

# This program generates an inverted index that maps n-grams tokens to 
# paths. The index is called "token_path_dict". 

# FIRST ARGUMENT: the root directory path.
# SECOND ARGUMENT: the length of tokens cut from filenames.

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: a list of the paths of every file in the directory "path".

def DFS(path):
    stack = []
    return_list = []
    file_path_dict = {}
    filenames = []
    stack.append(path);
    while len(stack) > 0:
        tmp = stack.pop(len(stack) - 1)
        # if this is a valid path.
        if(os.path.isdir(tmp)):

            return_list.append(tmp)
            # for every item in the "tmp" directory
            for item in os.listdir(tmp):
                # throws the path given by "tmp" + "item" onto the stack
                # for each "item".
                stack.append(os.path.join(tmp, item))
        # if it's not a valid path but it IS a valid file.
        elif(os.path.isfile(tmp)):
            # put the file in "return_list"
            # NOTE that "tmp" will be the entire path of a file.
            return_list.append(tmp)
            # returns path format: '/home/ljung/shit/shitposting'
            # iterates backwards through the path
            # adds only the file name to "filenames"
            filename = ""
            for c in tmp[::-1]:
                if c=="/":
                    break
                filename = c+filename
            filenames.append(filename)
            # adds a dictionary key value pair that maps the filename
            # to it's path.
            file_path_dict.update({filename:tmp})

    return (file_path_dict, filenames)

#=========1=========2=========3=========4=========5=========6=========7=

# "filenames" is the list of all the file names
# "length" is the q or n value (length of each token)
# RETURNS: a list of strings of length "length" cut from the filename
# using a window which propagates one character at a time from the
# beginning of the filename with fillers added to the end.
def gen_tokens(filenames, length):
    file_token_dict = {}
    for filename in filenames:
        # generates a string of length "length - 1" of all "/"s
        filler = "".join(((length-1)*["/"]))
        orig_name = filename
        # makes "filename" look like "///<old_filename>///"
        filename = filler + filename + filler
        # we stop "length - 1" characters from the end of the filename
        # with the filler on each end so that we aren't appending
        # tokens which consist only of filler characters ("/"s)

        token_list = []
        for pos in range(0,len(filename) - (length - 1)):
            token_list.append(filename[pos:pos+length])
        file_token_dict.update({orig_name:token_list})
    return file_token_dict



# MAIN PROGRAM

root_path = sys.argv[1]
token_length_string = sys.argv[2]
token_length = int(token_length_string)

paths_and_names = DFS(root_path)
paths_dict = paths_and_names[0]
filenames = paths_and_names[1]

# "file_token_dict" maps filenames to the tokens
file_token_dict = gen_tokens(filenames,token_length)

token_path_dict = {}

for filename in filenames:
    # the path of the filename
    path = paths_dict.get(filename)
    # the list of tokens associate with that file
    token_list = file_token_dict.get(filename)
    for token in token_list:
        # list of existing paths associated with given token
        path_list = token_path_dict.get(token)
        # if there are no existing paths, add one
        # if there are existing paths,
        # check if "path" is in "path_list", if not, append "path"
        if (path_list == None):
            path_list = [path]
        elif path not in path_list:
            path_list.append(path)
        # update the dictionary with the new path list
        token_path_dict.update({token:path_list})



for key in token_path_dict:
    print(key,token_path_dict.get(key))


#=========1=========2=========3=========4=========5=========6=========7=

import sys
import os



# FIRST ARGUMENT: the root directory path.
# SECOND ARGUMENT: the length of tokens cut from filenames.

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: a list of the paths of every file in the directory "path".
# calls "gen_tokens" to generate tokens from each pathname

def DFS(path, token_length):
    stack = []
    file_path_dict = {}
    file_pathtokens_dict = {}
    filenames = []
    stack.append(path);
    while len(stack) > 0:
        tmp = stack.pop(len(stack) - 1)
        tmp2 = tmp[17:]
        # if this is a valid path.
        if(os.path.isdir(tmp)):
            # for every item in the "tmp" directory
            for item in os.listdir(tmp):
                # throws the path given by "tmp" + "item" onto the stack
                # for each "item".
                stack.append(os.path.join(tmp, item))
        # if it's not a valid path but it IS a valid file.
        elif(os.path.isfile(tmp)):
            # put the file in "return_list"
            # NOTE that "tmp" will be the entire path of a file.
            # returns path format: '/home/ljung/shit/shitposting'
            filename = ""
            for c in tmp2[::-1]:
                if c=="/":
                    break
                filename = c+filename
            filenames.append(filename)
            # adds a dictionary pair mapping filename to path tokens
            file_pathtokens_dict.update({filename:gen_tokens(tmp2,token_length)})
            # adds a dictionary pair mapping filename to its path
            file_path_dict.update({filename:tmp2})

    return (file_pathtokens_dict, file_path_dict)

#=========1=========2=========3=========4=========5=========6=========7=

# "pathname" is one  path
# "length" is the q or n value (length of each token)
# RETURNS: a list of strings of length "length" cut from the path
# using a window which propagates one character at a time from the
# beginning of the filename with fillers added to the end.
def gen_tokens(pathname, length):
    file_token_dict = {}
    # generates a string of length "length - 1" of all "/"s
    filler = "".join(((length-1)*["|"]))
    orig_name = pathname
    # makes "pathname" look like "|||<old_pathname>|||"
    pathname = filler + pathname + filler
    # we stop "length - 1" characters from the end of the filename
    # with the filler on each end so that we aren't appending
    # tokens which consist only of filler characters ("|"s)

    token_list = []
    for pos in range(0,len(pathname) - (length - 1)):
        token_list.append(pathname[pos:pos+length])
    return token_list



def get_all_paths(input):
    f = open("paths.txt", "w")   
    for key, value in input.items():
        f.write(value + "\n") 
    f.close()    


# MAIN PROGRAM

root_path = sys.argv[1]
token_length = int(sys.argv[2])

allpaths = DFS(root_path, token_length)
#print(allpaths)

os.chdir("/home/ljung/CDIAC-clust")

file_pathtokens_dict = allpaths[0]
file_path_dict = allpaths[1]
get_all_paths(file_path_dict)

f1 = open(root_path + "file_pathtokens_dict2.txt","w")
f2 = open(root_path + "file_path_dict2.txt","w")

f1.write( str(file_pathtokens_dict) )
f2.write( str(file_path_dict) )
f1.close()
f2.close() 

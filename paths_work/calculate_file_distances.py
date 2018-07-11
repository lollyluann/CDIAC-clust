import os
import sys

def path_dist(path1, path2):
    path1_folders = path1.split("/")
    path2_folders = path2.split("/")
    # "shared" contains the shared directory closest to the bottom
    # "common_index" contains the index of "shared"
    shared = ""
    common_index = 0
    for i in range(min(len(path1_folders,path2_folders))):
        if path1_folders[i]==path2_folders[i]:
            shared = path1_folders[i]
            common_index = i
        else:
            break
    
    p1_dist_to_shared = len(path1_folders)-1-common_index
    p2_dist_to_shared = len(path2_folders)-1-common_index

    total_dist = p1_dist_to_shared + p2_dist_to_shared
    return total_dist

def find_max_dist(root):
    stack = []
    count = 1
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

def find_max_dist(root):
    # "initial_depth" is the depth until siblings are reached
    initial_depth = 0
    stack = []
    stack.append(root)
    while len(stack) > 0:
        tmp = stack.pop(len(stack)-1)
        children = []
        for item in os.listdir(tmp):
            if os.path.isdir(item):
                children.append(os.path.join(tmp, item))
                stack.append(os.path.join(tmp, item))
        if len(children)<2:
            continue
        else:
            #do DFS on each child
            break
        
            

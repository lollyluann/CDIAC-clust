import os
import sys

'''commit_msg = sys.argv[1]
counter = 0
waste = ""
for a in sys.argv:
    if counter>1:
        commit_msg = commit_msg + " " + a
    else:
        waste = a
    counter += 1
'''

commit_msg = sys.argv[1]
'''for a in sys.argv:
    commit_msg = commit_msg + " " + a'''


os.chdir("/home/ljung/CDIAC-clust")
os.system("git add .")
commit_cmd = "git commit -m \'" + commit_msg + "\'"
os.system(commit_cmd)
os.system("git push")
print("Committed successfully!")

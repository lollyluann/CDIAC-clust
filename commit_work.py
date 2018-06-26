import os
import sys

commit_msg = ""
for a in sys.argv:
    commit_msg = commit_msg + a
os.chdir("/home/ljung/CDIAC-clust")
os.system("git add .")
commit_cmd = "git commit -m" + commit_msg
os.system(commit_cmd)
os.system("git push")
print("Committed successfully!")

import os
import sys

commit_msg = sys.argv[1]

os.chdir("/home/ubuntu/cluster-datalake")
os.system("git add .")
commit_cmd = "git commit -m \'" + commit_msg + "\'"
os.system(commit_cmd)
os.system("git push")
print("Committed successfully!")

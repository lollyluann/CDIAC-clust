%%time 

import locale
import glob
import os.path
import requests
import tarfile
import sys
import codecs
from smart_open import smart_open
import re

dirname = 'aclImdb'
filename = 'aclImdb_v1.tar.gz'
locale.setlocale(locale.LC_ALL, 'C')
all_lines = []

if sys.version > '3':
    control_chars = [chr(0x85)]
else:
    control_chars = [unichr(0x85)]

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", norm_text)
    return norm_text

if not os.path.isfile('aclImdb/alldata-id.txt'):
    if not os.path.isdir(dirname):
        if not os.path.isfile(filename):
            # Download IMDB archive
            print("Downloading IMDB archive...")
            url = u'http://ai.stanford.edu/~amaas/data/sentiment/' + filename
            r = requests.get(url)
            with smart_open(filename, 'wb') as f:
                f.write(r.content)
        # if error here, try `tar xfz aclImdb_v1.tar.gz` outside notebook, then re-run this cell
        tar = tarfile.open(filename, mode='r')
        tar.extractall()
        tar.close()
    else:
        print("IMDB archive directory already available without download.")

    # Collect & normalize test/train data
    print("Cleaning up dataset...")
    folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']
    for fol in folders:
        temp = u''
        newline = "\n".encode("utf-8")
        output = fol.replace('/', '-') + '.txt'
        # Is there a better pattern to use?
        txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
        print(" %s: %i files" % (fol, len(txt_files)))
        with smart_open(os.path.join(dirname, output), "wb") as n:
            for i, txt in enumerate(txt_files):
                with smart_open(txt, "rb") as t:
                    one_text = t.read().decode("utf-8")
                    for c in control_chars:
                        one_text = one_text.replace(c, ' ')
                    one_text = normalize_text(one_text)
                    all_lines.append(one_text)
                    n.write(one_text.encode("utf-8"))
                    n.write(newline)

    # Save to disk for instant re-use on any future runs
    with smart_open(os.path.join(dirname, 'alldata-id.txt'), 'wb') as f:
        for idx, line in enumerate(all_lines):
            num_line = u"_*{0} {1}\n".format(idx, line)
            f.write(num_line.encode("utf-8"))

assert os.path.isfile("aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"
print("Success, alldata-id.txt is available for next steps.")

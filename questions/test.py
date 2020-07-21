import sys
import os
directory='corpus'
files={}
for filname in os.listdir(directory):
    with open(os.path.join(directory,filname)) as f:
        content=''.join([letter for letter in f.read()])
        files[filname]=content
print(files)
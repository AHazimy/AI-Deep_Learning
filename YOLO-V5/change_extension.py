import os

dir =("annotations")
for i in os.listdir(dir):
    files = os.path.join(dir,i)
    split= os.path.splitext(files)
    if split[1]=='.jpg':
       os.rename(files,split[0]+'.txt')
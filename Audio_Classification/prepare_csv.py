import pandas as pd 
import os
from pathlib import Path


path = "audio//"
 
dir_list = os.listdir(path)
 
df_list = []

for (root, dirs, file) in os.walk(path):

    for f in file:
        
        if '.wav' in f:
            df_list.append([Path(os.path.join(root,f)).absolute(), os.path.split(root)[-1]])
            
df = pd.DataFrame(df_list, columns=["file_name", "class"])

print(df.head(10))

df.to_csv('myData.csv')

import os 
import math

train_data_path = os.path("./data/train/")
test_data_path = os.path("./data/test/")

for dr in os.listdir(train_data_path):
    if dr.isdir():
        _, _, pics = next(os.walk(os.path.join(train_data_path, dr)))
        pics_count = len(pics)
        
        all_pics = os.listdir(dr)
        
        for i in range(math.ceil(pics_count/4)):
            abs_path = os.path.abspath(all_pics[i])
            os.rename(abs_path, dr)
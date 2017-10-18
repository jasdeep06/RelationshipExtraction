import numpy as np
import re

f=open("dataset/TRAIN_FILE.txt")
text=f.read()
splitted=re.split('\n',text)
for i in range(0,32000,4):
    nest_split=splitted[i].split()
    nest_split=nest_split[1:]
    print(" ".join(nest_split))

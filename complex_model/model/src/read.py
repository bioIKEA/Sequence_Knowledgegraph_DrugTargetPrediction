import numpy as np
import codecs
import pandas as pd

 
path = '../data'

f = codecs.open(path + './new_data/smile.txt', mode='r', encoding='utf-8')
line = f.readline()
drug_data = []

while line:
    a = line.split()
    b = a[1:2]
    drug_data.append(b)
    line = f.readline()

f.close()

drug = [str(x) for item in drug_data for x in item]
drug = list(drug)

print(len(drug))

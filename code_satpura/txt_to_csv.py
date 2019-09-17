import pandas as pd

# f=open('/home/Drive2/DATA_LUNG_TCGA/data/train/adeno','r')
# for line in f.readline():
import os
directory ='/home/Drive2/DATA_LUNG_TCGA/data/train/adeno'
for filename in os.listdir(directory):
	txt = list(open(filename))
	wsi_path= txt[0][:-1]
	level, patch = map(int, txt[1][:-1].split())
	x, y = map(int, txt[h][:-1].split('__'))
	prediction = pd.DataFrame({'filename':filename, 'wsi_path':wsi_path,'patch':patch,'level':level,'x':x,'y':y}).to_csv('data_adeno.csv',mode='a', header=True)


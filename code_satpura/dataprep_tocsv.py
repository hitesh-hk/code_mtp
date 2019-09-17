import os
import pandas as pd
rootdir ='/home/Drive2/DATA_LUNG_TCGA/data/val'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
		#print os.path.join(subdir, file)
		filepath = subdir + os.sep + file

		# if filepath.endswith(".asm"):
		# print (filepath)
		txt = list(open(filepath))
		wsi_path= txt[0][:-1]
		level, patch = map(int, txt[1][:-1].split())
		if ("data_adeno" in wsi_path):
			label = 0
		if ("data_squamous" in wsi_path):
			label = 1
		# print(label)
		for h in range(2,len(txt)):
			x, y = map(int, txt[h][:-1].split('__'))
			prediction = pd.DataFrame({'wsi_path':wsi_path,'patch':patch,'level':level,'x':x,'y':y,'label':label},index=[h]).to_csv('data_val.csv',mode='a', header=False)

import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.DataFrame(np.random.random((7,7)),columns=['a','b','c','d','e','f','g'])
#sn.heatmap(df)
sn.heatmap(df,annot=True,annot_kws={'size':7},xticklabels=False)
plt.show()

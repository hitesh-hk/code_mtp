#import tkinter
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
# Generate some test data
x = np.random.randn(8873)
y = np.random.randn(8873)

sns.jointplot(x=x, y=y, kind='hex')
plt.show()

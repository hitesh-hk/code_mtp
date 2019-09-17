import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from skimage.color import rgb2gray
import matplotlib as mpl
# Read original image
# img = io.imread('img.jpg')

# # Get the dimensions of the original image
# x_dim, y_dim, z_dim = np.shape(img)

# Read CSV with a Pandas DataFrame
df = pd.read_csv("prediction.csv")

# Get the dimensions of the original image
x_dim = np.int(df["wsi_dimensions_0"][1])/10
y_dim = np.int(df["wsi_dimensions_1"][1])/10

# Create heatmap
heatmap = np.zeros((x_dim, y_dim), dtype=float)
print(heatmap.shape)
# exit()

# Set probabilities values to specific indexes in the heatmap
for index, row in df.iterrows():
    x = np.int(row["x"]/10)
    y = np.int(row["y"]/10)
    patch = np.int(row["patch"]/10)
    # y1 = np.int(row["y1"])
    # p = row["probabilities"]
    p = row["probability_0"]
    q = row["probability_1"]
    if(p>q):
        heatmap[x:x+patch,y:y+patch] = p
    else:
        heatmap[x:x+patch,y:y+patch] = q




plt.imshow(heatmap)
cbar = plt.colorbar()
cbar.ax.set_ylabel('probabilities')
plt.imsave("gen_heatmap.png",heatmap)
plt.show()


# # Plot images
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# ax = axes.ravel()
# img=heatmap
# plt.imshow(img)
# ax[0].set_title("Original")
# fig.colorbar(ax[0].imshow(img), ax=ax[0])

# # # ax[1].imshow(img, vmin=0, vmax=1)
# # ax[1].imshow(heatmap, alpha=.5, cmap='jet')
# # ax[1].set_title("Original + heatmap")

# # Specific colorbar
# norm = mpl.colors.Normalize(vmin=0,vmax=2)
# N = 11
# cmap = plt.get_cmap('jet',N)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# plt.colorbar(sm, ticks=np.linspace(0,1,N), 
#              boundaries=np.arange(0,1.1,0.1)) 

# fig.tight_layout()
# plt.show()

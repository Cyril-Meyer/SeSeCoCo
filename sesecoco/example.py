import numpy as np
import skimage
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.colors

import sesecoco

print("SeSeCoCo : Semantic Segmentation Comparator using Components")

seg1 = skimage.io.imread("example/seg1.png")
seg2 = skimage.io.imread("example/seg2.png")

seg1 = (seg1 == 255)*1
seg2 = (seg2 == 255)*1

# Tests
result1 = sesecoco.cmp(seg1, seg2, 0.40, 0.40, True)
result2 = sesecoco.cmp(seg1, seg2, 0.40, 0.40, True)
result3 = sesecoco.cmp(seg1, seg2, 0.80, 0.80, True)

print(result1)
print(result2)
print(result3)
print(result1 + result2)
# The following line will not work
# print(result1 + result3)

# Graphical view of segmentation difference
mpl_green = matplotlib.colors.ListedColormap([0, 1, 0, 1])

mpl_red = matplotlib.colors.ListedColormap([1, 0, 0, 1])
mpl_orange = matplotlib.colors.ListedColormap([1, 0.5, 0, 1])
mpl_yellow = matplotlib.colors.ListedColormap([1, 1, 0, 1])

mpl_blue = matplotlib.colors.ListedColormap([0, 0, 1, 1])
mpl_cyan = matplotlib.colors.ListedColormap([0, 1, 1, 1])

plt.imshow(np.ones(seg1.shape), cmap='Paired')
plt.imshow(np.ma.masked_where(result1.seg_un == 0, result1.seg_tp), cmap=mpl_cyan)
plt.imshow(np.ma.masked_where(result1.seg_un_d == 0, result1.seg_tp), cmap=mpl_orange)
plt.imshow(np.ma.masked_where(result1.seg_ov == 0, result1.seg_tp), cmap=mpl_yellow)
plt.imshow(np.ma.masked_where(result1.seg_fn == 0, result1.seg_tp), cmap=mpl_blue)
plt.imshow(np.ma.masked_where(result1.seg_fp == 0, result1.seg_tp), cmap=mpl_red)
plt.imshow(np.ma.masked_where(result1.seg_tp == 0, result1.seg_tp), cmap=mpl_green)

plt.show()

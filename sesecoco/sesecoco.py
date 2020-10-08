import numpy as np
import skimage
import skimage.io
import skimage.measure
import matplotlib.pyplot as plt

# user parameters
param_minimum_recall_tp = 0.40
param_minimum_precision_fp = 0.40

# results
tp = 0
fp = 0
fn = 0

print("SeSeCoCo : Semantic Segmentation Comparator using Components")

seg1 = skimage.io.imread("example/seg1.png")
seg2 = skimage.io.imread("example/seg2.png")

if not (seg1.shape == seg2.shape):
    print("ERROR: segmentation shape not equal")
    exit(1)

seg1 = (seg1 == 255)*1
seg2 = (seg2 == 255)*1

# The argument 'neighbors' is deprecated, use 'connectivity' instead. For neighbors=8, use connectivity=2.
seg1_labels = skimage.measure.label(seg1, connectivity=2)
seg2_labels = skimage.measure.label(seg2, connectivity=2)
seg1_n_cc = seg1_labels.max()
seg2_n_cc = seg2_labels.max()

seg_tp = np.zeros(seg1.shape, dtype=np.uint8)
seg_fn = np.zeros(seg1.shape, dtype=np.uint8)
seg_fp = np.zeros(seg1.shape, dtype=np.uint8)


for cc_label in range(1, seg1_n_cc+1):
    # the current connected component
    current_cc = ((seg1_labels == cc_label)*1).astype(np.uint8)
    # the union of connected components which have a non null intersection with current_cc
    intersected_cc = np.zeros(seg1.shape, dtype=np.uint8)

    for cc2_label in range(1, seg2_n_cc+1):
        current_cc2 = (seg2_labels == cc2_label)*1
        if np.sum(current_cc * current_cc2) > 0:
            intersected_cc = intersected_cc + current_cc2

    recall = np.sum(current_cc * intersected_cc) / np.sum(current_cc)
    if recall > param_minimum_recall_tp:
        tp += 1
        seg_tp += current_cc
    else:
        fn += 1
        seg_fn += current_cc

for cc_label in range(1, seg2_n_cc+1):
    # the current connected component
    current_cc = ((seg2_labels == cc_label) * 1).astype(np.uint8)
    # the union of connected components which have a non null intersection with current_cc
    intersected_cc = np.zeros(seg1.shape, dtype=np.uint8)

    for cc1_label in range(1, seg1_n_cc+1):
        current_cc1 = (seg1_labels == cc1_label)*1
        if np.sum(current_cc * current_cc1) > 0:
            intersected_cc = intersected_cc + current_cc1

    precision = np.sum(current_cc * intersected_cc) / np.sum(current_cc)
    if precision < param_minimum_precision_fp:
        fp += 1
        seg_fp += current_cc

'''
plt_result = seg_tp*3 + 5*seg_fn + 7*seg_fp
plt.imshow(plt_result, cmap='Paired', vmin=0, vmax=12)
plt.show()
'''

print("Segmentation 1 (y_true)")
print("Connected Components", seg1_n_cc)
print("Correctly detected components", tp)
print("Not detected components (too small or not enough recall)", fn)
print("Segmentation 2 (y_pred)")
print("Connected Components", seg2_n_cc)
print("Over detecting components (too large or not enough precision)", fp)

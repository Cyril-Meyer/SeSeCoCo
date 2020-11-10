import numpy as np
import skimage
import skimage.measure


class SeSeCoCoResult:
    def __init__(self, minimum_recall_tp=0.50, minimum_precision_fp=0.50, with_segmentations=False):
        # user parameters
        self.param_minimum_recall_tp = minimum_recall_tp
        self.param_minimum_precision_fp = minimum_precision_fp
        # result
        self.seg1_n_cc = 0
        self.seg2_n_cc = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.over_detecting = 0
        self.under_detected = 0
        # optional results
        self.keep_segmentations = with_segmentations
        self.seg_tp = None
        self.seg_fn = None
        self.seg_fp = None
        self.seg_ov = None
        self.seg_un = None
        self.seg_un_d = None

    def __str__(self):
        return ("Segmentation 1 (y_true) " + "\n"
                "Connected Components " + str(self.seg1_n_cc) + "\n"
                "Correctly detected components " + str(self.tp) + "\n"
                "False negative components " + str(self.fn) + "\n"
                "Under detected components (too small or not enough recall) " + str(self.under_detected) + "\n"
                "Segmentation 2 (y_pred) " + "\n"
                "Connected Components " + str(self.seg2_n_cc) + "\n"
                "False positive components " + str(self.fp) + "\n"
                "Over detecting components (too large or not enough precision) " + str(self.over_detecting) + "\n")

    def get_score(self):
        return self.seg1_n_cc, self.tp, self.fn, self.under_detected, self.seg2_n_cc, self.fp, self.over_detecting

    def __add__(self, other):
        if not other.__class__ is SeSeCoCoResult:
            print("ERROR: both arguments must be SeSeCoCoResult")
            return NotImplemented

        if not (self.param_minimum_recall_tp == other.param_minimum_recall_tp and
                self.param_minimum_precision_fp == other.param_minimum_precision_fp):
            print("ERROR: parameters (recall and precision) must be the same")
            return NotImplemented

        result = SeSeCoCoResult()

        result.seg1_n_cc = self.seg1_n_cc + other.seg1_n_cc
        result.seg2_n_cc = self.seg2_n_cc + other.seg2_n_cc
        result.tp = self.tp + other.tp
        result.fp = self.fp + other.fp
        result.fn = self.fn + other.fn
        result.over_detecting = self.over_detecting + other.over_detecting
        result.under_detected = self.under_detected + other.under_detected

        result.keep_segmentations = False
        result.seg_tp = None
        result.seg_fn = None
        result.seg_fp = None
        result.seg_ov = None
        result.seg_un = None
        result.seg_un_d = None

        return result


def cmp(seg1, seg2, minimum_recall_tp=0.50, minimum_precision_fp=0.50, with_segmentations=False):
    # user parameters
    param_minimum_recall_tp = minimum_recall_tp
    param_minimum_precision_fp = minimum_precision_fp

    # results
    tp = 0
    fp = 0
    fn = 0
    over_detecting = 0
    under_detected = 0

    if not (seg1.shape == seg2.shape):
        print("ERROR: segmentation shape not equal")
        return 1

    # The argument 'neighbors' is deprecated, use 'connectivity' instead. For neighbors=8, use connectivity=2.
    seg1_labels = skimage.measure.label(seg1, connectivity=2)
    seg2_labels = skimage.measure.label(seg2, connectivity=2)
    seg1_n_cc = seg1_labels.max()
    seg2_n_cc = seg2_labels.max()

    seg_tp = np.zeros(seg1.shape, dtype=np.uint8)
    seg_fn = np.zeros(seg1.shape, dtype=np.uint8)
    seg_fp = np.zeros(seg1.shape, dtype=np.uint8)
    seg_ov = np.zeros(seg1.shape, dtype=np.uint8)
    seg_un = np.zeros(seg1.shape, dtype=np.uint8)
    seg_un_d = np.zeros(seg1.shape, dtype=np.uint8)

    for cc_label in range(1, seg1_n_cc + 1):
        # the current connected component
        current_cc = ((seg1_labels == cc_label) * 1).astype(np.uint8)
        # the union of connected components which have a non null intersection with current_cc
        intersected_cc = np.zeros(seg1.shape, dtype=np.uint8)
        detect_cc = np.zeros(seg1.shape, dtype=np.uint8)

        for cc2_label in range(1, seg2_n_cc + 1):
            current_cc2 = (seg2_labels == cc2_label) * 1
            if np.sum(current_cc * current_cc2) > 0:
                intersected_cc = intersected_cc + current_cc2
                detect_cc += current_cc2.astype(np.uint8)

        recall = np.sum(current_cc * intersected_cc) / np.sum(current_cc)
        if np.sum(intersected_cc) == 0:
            fn += 1
            seg_fn += current_cc
        elif recall > param_minimum_recall_tp:
            tp += 1
            seg_tp += current_cc
        else:
            under_detected += 1
            seg_un += current_cc
            seg_un_d += detect_cc

    for cc_label in range(1, seg2_n_cc + 1):
        # the current connected component
        current_cc = ((seg2_labels == cc_label) * 1).astype(np.uint8)
        # the union of connected components which have a non null intersection with current_cc
        intersected_cc = np.zeros(seg1.shape, dtype=np.uint8)

        for cc1_label in range(1, seg1_n_cc + 1):
            current_cc1 = (seg1_labels == cc1_label) * 1
            if np.sum(current_cc * current_cc1) > 0:
                intersected_cc = intersected_cc + current_cc1

        precision = np.sum(current_cc * intersected_cc) / np.sum(current_cc)

        if np.sum(intersected_cc) == 0:
            fp += 1
            seg_fp += current_cc
        elif precision < param_minimum_precision_fp:
            over_detecting += 1
            seg_ov += current_cc

    result = SeSeCoCoResult(minimum_recall_tp, minimum_precision_fp, with_segmentations)
    result.seg1_n_cc = seg1_n_cc
    result.seg2_n_cc = seg2_n_cc
    result.tp = tp
    result.fp = fp
    result.fn = fn
    result.over_detecting = over_detecting
    result.under_detected = under_detected

    if with_segmentations:
        result.seg_tp = seg_tp
        result.seg_fn = seg_fn
        result.seg_fp = seg_fp
        result.seg_ov = seg_ov
        result.seg_un = seg_un
        result.seg_un_d = seg_un_d

    return result


def fast_cmp(seg1, seg2, minimum_recall_tp=0.50, minimum_precision_fp=0.50):
    # user parameters
    param_minimum_recall_tp = minimum_recall_tp
    param_minimum_precision_fp = minimum_precision_fp

    # results
    tp = 0
    fp = 0
    fn = 0
    over_detecting = 0
    under_detected = 0

    if not (seg1.shape == seg2.shape):
        print("ERROR: segmentation shape not equal")
        return 1

    # The argument 'neighbors' is deprecated, use 'connectivity' instead. For neighbors=8, use connectivity=2.
    seg1_labels = skimage.measure.label(seg1, connectivity=2)
    seg2_labels = skimage.measure.label(seg2, connectivity=2)
    seg1_n_cc = seg1_labels.max()
    seg2_n_cc = seg2_labels.max()

    for cc_label in range(1, seg1_n_cc + 1):
        # the current connected component
        current_cc = ((seg1_labels == cc_label) * 1).astype(np.uint8)
        # the union of connected components which have a non null intersection with current_cc
        intersected_cc = current_cc * seg2
        recall = np.sum(current_cc * intersected_cc) / np.sum(current_cc)

        if np.sum(intersected_cc) == 0:
            fn += 1
        elif recall > param_minimum_recall_tp:
            tp += 1
        else:
            under_detected += 1

    for cc_label in range(1, seg2_n_cc + 1):
        # the current connected component
        current_cc = ((seg2_labels == cc_label) * 1).astype(np.uint8)
        # the union of connected components which have a non null intersection with current_cc
        intersected_cc = current_cc * seg1
        precision = np.sum(current_cc * intersected_cc) / np.sum(current_cc)

        if np.sum(intersected_cc) == 0:
            fp += 1
        elif precision < param_minimum_precision_fp:
            over_detecting += 1

    result = SeSeCoCoResult(minimum_recall_tp, minimum_precision_fp, False)
    result.seg1_n_cc = seg1_n_cc
    result.seg2_n_cc = seg2_n_cc
    result.tp = tp
    result.fp = fp
    result.fn = fn
    result.over_detecting = over_detecting
    result.under_detected = under_detected

    return result

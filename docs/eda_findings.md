# GestureFlow exploratory findings

The static dataset has 15,029 examples across five gestures. We found zero missing values and zero exact duplicate rows. The coordinates range from −1 to 1. Removing the two constant wrist coordinates leaves 40 features.

The training set contains 2,400 Open, 2,426 Close, 2,374 Pointer, 2,397 Ok and 2,500 Nice examples. The test set contains the remaining 2,932 examples. The average absolute coordinate difference between consecutive same-class rows was 0.0164, compared with 0.1584 for randomly paired rows of the same class. Keeping chunks of 100 adjacent rows together reduces the risk of testing on nearly identical frames. Nearby chunks may still come from the same recording.

The mean hand skeletons show an open palm, a closed fist, an extended index finger, an OK ring and a thumbs-up (Nice). Ok has a median thumb-to-index fingertip distance of 0.074. The other four medians range from 0.312 to 0.601. Pointer has the shortest median pinky-to-wrist distance at 0.290.

Open and Ok have the closest average poses (Euclidean distance 1.19). The first two principal components explain 76.0% of the variation in the 4,000 plotted training examples. Open and Ok overlap in the lower-right band. The dense Nice cluster lies farther left, with some Nice and Pointer examples overlapping toward the middle. A two-dimensional overlap alone cannot establish classifier errors.

Neighbouring joints on the same finger often move together. In the training data, the horizontal coordinates of the index fingertip and its nearest joint have a Pearson correlation of 0.958. The corresponding vertical coordinates on the pinky have a correlation of 0.979.

# First Place Solution of AIDAO24 Online Stage

## Task 1
We divide the dataset into **two atlases** so that the features are from the same domain. We build a correspondence from **one atlas to another**, searching for the most similar samples in different atlases. The **similarity** of the samples is determined by the degree of correlation of the features.

Having obtained an unambiguous correspondence between samples in different atlases, we cluster them in the **Brainnetome** atlas, discarding features with an index of 210 and higher, since it was worse with them. We build **Connectivity Matrixes** (hereinafter CM) from the Nilearn library for samples using the `kind="precision"` parameter and using connectivity from the other hemisphere for features from one hemisphere (so clustering is performed unambiguously).

Having received CM, we pull them into a **vector** and use them to determine the similarity of samples among themselves, counting the **Pearson correlation** between the obtained vectors. Next, we build a **graph** with weights for clustering.

## Task 2
- The data was **augmented**: the time line of the images was divided into sub-sections of length 80 in increments of 40
- We did **undersampling**: according to the Beijing dataset, in order to withstand **50/50** class balancing everywhere
- Training: a **LogReg** with **L1** regularization is pre-trained, after which unimportant features are cut off (where weights <= 1e-5); then we reduce the dimension using **PCA** and train another **LogReg** on the final data set

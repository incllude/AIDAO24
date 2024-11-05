import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from scripts.data_utils import get_connectome
from scripts.classification_models import LogRegPCAL1 
import pickle

import warnings
warnings.filterwarnings('ignore')

np.random.seed(12345)


# load both files with series

bnu_series_path = '../data/ts_cut/HCPex/bnu{}.npy'
bnu_labels_path = '../data/ts_cut/HCPex/bnu.csv'
ihb_series_path = '../data/ts_cut/HCPex/ihb.npy'
ihb_labels_path = '../data/ts_cut/HCPex/ihb.csv'

X_bnu = np.concatenate([np.load(bnu_series_path.format(i)) for i in (1, 2)], axis=0)

X_bnu_sw = []
for i in range(0, X_bnu.shape[1] - 80 + 1, 40):
    X_bnu_sw.append(X_bnu[:, i : i+80])
X_bnu_sw = np.stack(X_bnu_sw, axis=1)

Y_bnu = pd.read_csv(bnu_labels_path)
Y_bnu = Y_bnu.to_numpy()
Y_bnu_sw = np.repeat(Y_bnu, X_bnu_sw.shape[1], axis=0)
print(Y_bnu_sw.shape)
X_bnu_sw = X_bnu_sw.reshape(-1, 80, 419)
print(X_bnu_sw.shape)


X_ihb = np.load(ihb_series_path)

X_ihb_sw = []
for i in range(0, X_ihb.shape[1] - 80 + 1, 40):
    X_ihb_sw.append(X_ihb[:, i : i+80])
X_ihb_sw = np.stack(X_ihb_sw, axis=1)

Y_ihb = pd.read_csv(ihb_labels_path)
Y_ihb = Y_ihb.to_numpy()
Y_ihb_sw = np.repeat(Y_ihb, X_ihb_sw.shape[1], axis=0)
print(Y_ihb_sw.shape)
X_ihb_sw = X_ihb_sw.reshape(-1, 80, 419)
print(X_ihb_sw.shape)

# time series have different length
# by the way ``get_connectome`` reduces them to matrices 419x419

X_bnu_sw = get_connectome(X_bnu_sw)
X_ihb_sw = get_connectome(X_ihb_sw)
X_bnu = get_connectome(X_bnu)
X_ihb = get_connectome(X_ihb)

# concat the train data
X_bnu = np.concatenate([X_bnu_sw, X_bnu])
Y_bnu = np.concatenate([Y_bnu_sw, Y_bnu])
X_ihb = np.concatenate([X_ihb_sw, X_ihb])
Y_ihb = np.concatenate([Y_ihb_sw, Y_ihb])

# Perform undersampling to balance the classes in Y_bnu
class_1_indices = np.where(Y_bnu == 1)[0]  # Indices of class 1
n_class_1 = len(class_1_indices)

# Randomly select indices from class 0
class_0_indices = np.where(Y_bnu == 0)[0]  # Indices of class 0
class_0_indices_sampled = np.random.choice(class_0_indices, size=n_class_1, replace=False)

# Combine the selected indices
balanced_indices = np.concatenate([class_1_indices, class_0_indices_sampled])

# Create balanced datasets
X_bnu_balanced = X_bnu[balanced_indices]
Y_bnu_balanced = Y_bnu[balanced_indices]

# Combine the balanced bnu data with ihb data
X_balanced = np.concatenate([X_bnu_balanced, X_ihb])
Y_balanced = np.concatenate([Y_bnu_balanced, Y_ihb])  # Convert Y_ihb to numpy array

# Split the balanced data into train and validation sets
x_train, x_validate, y_train, y_validate = train_test_split(X_balanced, Y_balanced,
                                                            test_size=0.15, random_state=10, stratify = Y_balanced)

# Print shapes of the train and validation sets
print("Train shapes:", x_train.shape, y_train.shape)
print("Validation shapes:", x_validate.shape, y_validate.shape)

unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class distribution in y_train:", class_distribution)

unique, counts = np.unique(y_validate, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class distribution in y_validate:", class_distribution)


logreg = LogRegPCAL1()
logreg.model_final.set_params(**{'C' : 0.0005, 'random_state' : 44})
logreg.pca.set_params(**{'n_components': 0.8})

train_acc = logreg.model_training(x_train, y_train)

conf_mat, acc, f1 = logreg.model_testing(x_validate, y_validate)


# save model and weights 

pkl_filename = "./model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(logreg, file)
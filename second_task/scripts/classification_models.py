import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier


class LogRegPCA:
    def __init__(self, pca=True):
        self.pca = PCA() if pca else None
        self.model = LogisticRegression()

    def model_training(self, x, y):
        x = self.preprocess(x)

        if self.pca is not None:
            x = self.pca.fit_transform(x)

        print(f'Features after PCA: {x.shape}')
        
        self.model.fit(x, y)

        acc = self.model.score(x, y)
        print('Accuracy on train:', round(acc, 3))

        return acc

    def model_predict(self, x):
        x = self.preprocess(x)

        if self.pca is not None:
            x = self.pca.transform(x)

        y_pred = self.model.predict(x)
        return y_pred

    def model_testing(self, x, y):
        y_pred = self.model_predict(x)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        print('Accuracy on test:', round(acc, 3))
        print('F1 score on test:', round(f1, 3))
        cm = confusion_matrix(y, y_pred)

        return cm, acc, f1

    def preprocess(self, x):
        vecs = zscore(x, axis=0)

        for i in vecs:
            np.fill_diagonal(i, 0)

        vecs = vecs.reshape((x.shape[0], -1))

        return vecs
        

class LogRegPCAL1:
    def __init__(self, pca=True):
        self.pca = PCA() if pca else None
        self.model_initial = LogisticRegression(penalty='l1', solver='liblinear', random_state = 44)
        self.model_final = LogisticRegression()
        self.important_features = None

    def model_training(self, x, y):
        x = self.preprocess(x)

        self.model_initial.fit(x, y)
        acc_initial = self.model_initial.score(x, y)
        print('Initial accuracy on train:', round(acc_initial, 3))

        x = self.remove_low_weight_features(x)

        if self.pca is not None:
            x = self.pca.fit_transform(x)

        print(f'Features after PCA: {x.shape}')

        self.model_final.fit(x, y)
        acc_final = self.model_final.score(x, y)
        print('Final accuracy on train:', round(acc_final, 3))

        return acc_final

    def remove_low_weight_features(self, x, threshold=1e-5):

        weights = np.abs(self.model_initial.coef_[0])
        self.important_features = weights > threshold  

        print(f'Features remaining after L1 regularization: {np.sum(self.important_features)}/{len(weights)}')
        return x[:, self.important_features]

    def model_predict(self, x):
        x = self.preprocess(x)

        if self.important_features is not None:
            x = x[:, self.important_features]

        if self.pca is not None:
            x = self.pca.transform(x)

        y_pred = self.model_final.predict(x)
        return y_pred

    def model_testing(self, x, y):
        y_pred = self.model_predict(x)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        print('Accuracy on test:', round(acc, 3))
        print('F1 score on test:', round(f1, 3))
        cm = confusion_matrix(y, y_pred)

        return cm, acc, f1

    def preprocess(self, x):
        vecs = zscore(x, axis=0)

        for i in vecs:
            np.fill_diagonal(i, 0)

        vecs = vecs.reshape((x.shape[0], -1))

        return vecs

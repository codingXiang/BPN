import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self , data_path):
        self.df = pd.read_csv(data_path, header = None)
        self.output = self.one_hot_encoding(self.df.iloc[0:, 4].values)
        self.training_data = self.standard_deviation(self.df.iloc[0:, [0 , 1 , 2 , 3]].values)

    def standard_deviation(self , X):
        X_std = np.copy(X)
        for i in range(0 , X.shape[1]):
            X_std[: , i] = (X[: , i] - X[: , i].mean()) / X[: , i].std()
        return X_std
    def one_hot_encoding(self,Y):
        classes = np.unique(Y)
        number = [x for x in range(0 , classes.shape[0])]
        a = np.array([classes , number]).T
        for i in range(0 , a.shape[0]):
            Y = np.where(Y == a[i][0] , a[i][1] , Y)
        Y = [i for i in Y]
        targets = np.array(Y).reshape(-1)
        one_hot_targets = np.eye(a.shape[0])[targets]
        return one_hot_targets
    def output_process(self,Y):
        output = list()
        for i in Y:
            output.append(self.output_transform(i))
        return np.array(output)

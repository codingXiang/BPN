import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self , data_path):
        self.df = pd.read_csv(data_path, header = None)
        self.Y = self.one_hot_encoding(self.df.iloc[0:, 4].values)
        self.x = self.standard_deviation(self.df.iloc[0:, [0 , 1 , 2 , 3]].values)
    def split_data(self , x , Y , p):
        data = []
        for i in range(x.shape[0]):
            data.append([])
            data[i].append(np.array(x[i]))
            data[i].append(np.array(Y[i]))
        np.random.shuffle(data)

        split = int(Y.shape[0] * p)
        data = np.array(data)
        self.train_x  , self.train_Y = data[: split , 0] , data[: split , 1]
        self.test_x  , self.test_Y = data[split:  , 0] , data[split:  , 1]
        self.train_x = np.array([x.tolist() for x in self.train_x.tolist()])
        self.train_Y = np.array([Y.tolist() for Y in self.train_Y.tolist()])
        self.test_x = np.array([x.tolist() for x in self.test_x.tolist()])
        self.test_Y = np.array([Y.tolist() for Y in self.test_Y.tolist()])
        return self
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

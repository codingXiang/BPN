import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import Dataset
from diagram import Diagram
class BPNN(object):
    def __init__(self , dataset , learning_rate = 0.01 , n_iter = 10000 , momentum = 0.9 , shutdown_condition = 0.01):
        self.n_iter = n_iter
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.x = dataset.train_x
        self.Y = dataset.train_Y
        self.shutdown_condition = shutdown_condition
        self.cost = []
        self.momentum = momentum
        self.setup()
        self.diagram = Diagram(self)
    def setup(self):
        self.set_nn_architecture()
        self.set_weight()
    # step1
    def set_nn_architecture(self):
        self.input_node = self.x.shape[1]
        self.output_node = self.Y.shape[1]
        self.hidden_node = int((self.input_node + self.output_node) / 2)

        # bias
        self.h_b = np.random.random(self.hidden_node) * 0.3 + 0.1
        self.y_b = np.random.random(self.output_node) * 0.3 + 0.1
    # step2
    def set_weight(self):
        self.w1 = np.random.random((self.input_node , self.hidden_node))
        self.w2 = np.random.random((self.hidden_node , self.output_node))
    # step3
    def predict(self , x , Y):
        self.h = self.sigmoid((np.dot(x , self.w1) + self.h_b))
        self.y = self.sigmoid((np.dot(self.h , self.w2) + self.y_b))
        zy = np.where(self.y > 0.5 , 1 , 0)
        p_y = Y - zy
        self.acc = 0
        for i in p_y:
            if (i.sum() == 0):
                self.acc += 1
        self.acc = self.acc / Y.shape[0] * 100.0
        return self
    # step4
    def backend(self):
        E = (self.Y - self.y)
        errors = np.sum(np.square(E)) / self.Y.shape[1] / self.Y.shape[0]
        #### 輸出層 delta 計算
        delta_y = E * self.y * (1 - self.y)
        ### 隱藏層 delta 計算
        delta_h = (1 - self.h) * self.h * np.dot(delta_y , self.w2.T)
        # self.w2 += self.learning_rate * self.h.T.dot(delta_y) + self.momentum * self.h.T.dot(delta_y)
        # self.w1 += self.learning_rate * self.x.T.dot(delta_h) + self.momentum * self.x.T.dot(delta_h)
        self.w2 += self.learning_rate * self.h.T.dot(delta_y)
        self.w1 += self.learning_rate * self.x.T.dot(delta_h)
        self.y_b = self.learning_rate * delta_y.sum()
        self.h_b = self.learning_rate * delta_h.sum()
        return errors

    def train(self):
        self.error = 0
        for _iter in range(0 , self.n_iter):
            self.predict(self.x , self.Y)
            self.error = self.backend()
            self.cost.append(self.error)
            # if (_iter % 1000 == 0):
            #     print("Accuracy：%.2f" % self.acc)
            if (self.acc >= 98):
                return self
        return self

    def test(self):
        self.predict(self.dataset.test_x , self.dataset.test_Y)
        return self

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def draw(self , xlabel = '' , ylabel = '' , legend_loc = '' , title = ''):
        self.diagram.draw(xlabel , ylabel , legend_loc , title )

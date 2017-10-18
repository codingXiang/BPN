from bpnn import BPNN
from data import Dataset
from diagram import Diagram

if __name__ == "__main__":
    dataset = Dataset('iris.txt')
    bpn = BPNN(dataset , learning_rate = 0.01 , n_iter = 100000 , momentum = 0.9 , shutdown_condition = 0.01)
    bpn.train()
    bpn.test()
    print("Accuracy = %r %% , iteration = %r , MSE =  %.3f" %(bpn.acc , len(bpn.cost) , bpn.error))
    bpn.draw( title = 'Iris training process',
              xlabel = 'iterator numbers' ,
              ylabel = 'MSE',
              legend_loc = 'upper left')

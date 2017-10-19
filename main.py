from bpnn import BPNN
from data import Dataset
from diagram import Diagram

if __name__ == "__main__":
    dataset = Dataset('iris.txt')
    bpn = BPNN(dataset , learning_rate = 0.01 , n_iter = 10000 , momentum = 0.01 , shutdown_condition = 0.012)
    bpn.train()
    bpn.test()
    print("Accuracy = %r %% , iteration = %r , MSE =  %.3f" %(bpn.acc , len(bpn.cost) , bpn.error))
    bpn.draw( title = 'learning_rate = %r , MSE = %.3f , Accuracy = %.f %%' % (bpn.learning_rate , bpn.error , bpn.acc),
              xlabel = 'iterator numbers' ,
              ylabel = 'MSE',
              legend_loc = 'upper left')

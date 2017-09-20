import preprocess
import mlp, som, ga, ga_mlp
import numpy as np
import pylab as pl

# When this is set to true, the preprocessing of the data ignores two of the three sets of grass data. 
SIMPLE_GRASS = False
if SIMPLE_GRASS:
    LABEL_SIZE = 11
else:
    LABEL_SIZE = 13


def run_mlp():  
    '''
    Run a MLP on using pre-processed pollen data. 
    The percentage correct is stored in an array.
    '''         
    x = preprocess.preprocess('Pollens')
    pollen = np.array(x.create_one_file(SIMPLE_GRASS))
    pollen = x.normalise_max(pollen)
    train_set, train_set_target, test_set, test_set_target, validation_set, validation_set_target = x.make_groups(pollen, LABEL_SIZE, train_size=350, test_size=150, validation_size=150)
          
    p = mlp.mlp(train_set, train_set_target, 30, momentum = 0.9, outtype = 'softmax')
    error = p.earlystopping(train_set, train_set_target, validation_set, validation_set_target, 0.1, niterations=200)
    correct = p.confmat(test_set, test_set_target)

          
def run_som():
    '''
    Runs a SOM and outputs the best activations in a 2d grid. 
    Each class is given a unique symbol.
    '''
    x = preprocess.preprocess('Pollens')
    pollen = np.array(x.create_one_file(SIMPLE_GRASS))
    pollen = x.normalise_max(pollen)
    train_set, train_set_target, test_set, test_set_target, validation_set, validation_set_target = x.make_groups(pollen, LABEL_SIZE, algorithm='som', train_size=500, test_size=150, validation_size=0)

    net = som.som(20,20, train_set)
    net.somtrain(train_set, 400)
    best = np.zeros(np.shape(train_set)[0], dtype=int)
    for i in range(np.shape(train_set)[0]):
        best[i], activation = net.somfwd(train_set[i,:])
          
    pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
    where = pl.find(train_set_target == 0)    
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=15)
    where = pl.find(train_set_target == 1)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rv',ms=15)
    where = pl.find(train_set_target == 2)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'r^',ms=15)
    where = pl.find(train_set_target == 3)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'bs',ms=15)
    where = pl.find(train_set_target == 4)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'bv',ms=15)
    where = pl.find(train_set_target == 5)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=15)
    where = pl.find(train_set_target == 6)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gs',ms=15)
    where = pl.find(train_set_target == 7)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=15)
    where = pl.find(train_set_target == 8)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'g^',ms=15)
    where = pl.find(train_set_target == 9)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'ms',ms=15)
    where = pl.find(train_set_target == 10)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'mv',ms=15)
    where = pl.find(train_set_target == 11)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'m^',ms=15)
    where = pl.find(train_set_target == 12)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'ys',ms=15)
    pl.axis([-0.1,1.1,-0.1,1.1])
    pl.axis('off')
    pl.figure(2)

    best = np.zeros(np.shape(test_set)[0],dtype=int)
    for i in range(np.shape(test_set)[0]):
        best[i],activation = net.somfwd(test_set[i,:])
                
    pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
    where = pl.find(test_set_target == 0)    
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=15)
    where = pl.find(test_set_target == 1)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rv',ms=15)
    where = pl.find(test_set_target == 2)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'r^',ms=15)
    where = pl.find(test_set_target == 3)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'bs',ms=15)
    where = pl.find(test_set_target == 4)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'bv',ms=15)
    where = pl.find(test_set_target == 5)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=15)
    where = pl.find(test_set_target == 6)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gs',ms=15)
    where = pl.find(test_set_target == 7)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=15)
    where = pl.find(test_set_target == 8)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'g^',ms=15)
    where = pl.find(test_set_target == 9)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'ms',ms=15)
    where = pl.find(test_set_target == 10)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'mv',ms=15)
    where = pl.find(test_set_target == 11)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'m^',ms=15)
    where = pl.find(test_set_target == 12)
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'ys',ms=15)
    pl.axis([-0.1,1.1,-0.1,1.1])
    pl.axis('off')
    pl.show()

def run_som_pcn():
    '''
    Runs a perceptron using the activations from a SOM. 
    The initial data is split into two sets, one for use in the SOM, and the other for use in the perceptron. 
    '''
    x = preprocess.preprocess('Pollens')
    pollen = np.array(x.create_one_file(SIMPLE_GRASS))
    pollen = x.normalise_max(pollen)
    som_train_set, som_train_set_target, pcn_set, pcn_set_target, empty_set, empty_set_target = x.make_groups(pollen, LABEL_SIZE, algorithm='mlp', train_size=300, test_size=350, validation_size=0)
    net = som.som(5,5, som_train_set)
    net.somtrain(som_train_set, 300)
    net.run_perceptron(pcn_set, pcn_set_target, train_size=200, test_size=150)
             
def run_first_mlp_ga():
    '''
    GA with a fitness function for determining the optimum number of hidden neurons and what inputs should be activated or deactivated. 
    Mutation rate determined by dividing 1 by the fitness function string length. 
    '''
    pl.ion()
    pl.show()    
    x = ga.ga(84,'fF.fitness_mlp1',100,50,0.01,4,True)
    x.runGA()
        
def run_mlp_with_first_ga_results():
    '''
    Run the MLP with the results obtained from running run_first_mlp_ga.
    '''
    x = preprocess.preprocess('Pollens')
    pollen = np.array(x.create_one_file(SIMPLE_GRASS))
    pollen = x.normalise_max(pollen)
    train_set, train_set_target, test_set, test_set_target, validation_set, validation_set_target = x.make_groups(pollen, LABEL_SIZE, train_size=330, test_size=160, validation_size=160)
    hidden_neurons = 23
    weight_adjust = np.array([1,1,0,1,1,1,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0])
    
    
    p = ga_mlp.mlp(train_set, train_set_target, hidden_neurons, weight_adjust, momentum = 0.2, outtype = 'softmax')
    error = p.earlystopping(train_set, train_set_target, validation_set, validation_set_target, 0.1, niterations=200)
    correct = p.confmat(test_set, test_set_target)

def run_second_mlp_ga():
    '''
    GA with a fitness function for determining the optimum learning rate and number of iterations.
    Mutation rate determined by dividing 1 by the fitness function string length. 

    '''
    pl.ion()
    pl.show()    
    x = ga.ga(25,'fF.fitness_mlp2',50,50,0.05,4,True)
    x.runGA()


#run_mlp()            
#run_som()       
#run_som_pcn()    
#run_first_mlp_ga()
#run_mlp_with_first_ga_results()
#run_second_mlp_ga()
        
        
        
        
    
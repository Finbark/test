import numpy as np
import ga_mlp
import preprocess
import math

SIMPLE_GRASS = False
if SIMPLE_GRASS:
    LABEL_SIZE = 11
else:
    LABEL_SIZE = 13

def fitness_mlp1(pop):  
    '''
    Fitness function for determining the optimum number of hidden neurons and what inputs should be activated or deactivated in the MLP.
    '''  
    fitness = np.zeros((np.shape(pop)[0],1))
    for y in range (np.shape(pop)[0]):
        neurons_chunk, input_chunk = split_list1(pop[y]) 
        number_neurons = count_chunk(neurons_chunk)
        input_adjust = get_input_adjust(input_chunk)
        train_set, train_set_target, validation_set, validation_set_target = create_sets()
        score = run_mlp(train_set, train_set_target, number_neurons, validation_set, validation_set_target, 0.1, 200, input_adjust)    
        fitness[y] = score
    print_response_mlp1(fitness, pop) 
    fitness = np.squeeze(fitness)
    return fitness

def fitness_mlp2(pop):
    '''
    Fitness function for determining the optimum learning rate and number of iterations, uses the results obtained from running run_first_mlp_ga.
    The learning rate is a value between 0.05 and 0.5, while the number of iterations are between 100 and 400
    '''
    fitness = np.zeros((np.shape(pop)[0],1))
    hidden_neurons = 23
    weight_adjust = np.array([1,1,0,1,1,1,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0])
    for y in range (np.shape(pop)[0]):
        learning_rate, iterations = split_list2(pop[y]) 
        learning_rate = count_chunk(iterations) * 0.05 + 0.05 
        iterations = count_chunk(iterations) * 20 + 100
        train_set, train_set_target, validation_set, validation_set_target = create_sets()
        score = run_mlp(train_set, train_set_target, hidden_neurons, validation_set, validation_set_target, learning_rate, iterations, weight_adjust)    
        fitness[y] = score
    print_response_mlp2(fitness, pop) 
    fitness = np.squeeze(fitness)
    return fitness
    

def split_list1(pop_list):
    '''
    Splits the population data list into the portion corresponding to the number of neurons and the portion for the weight adjustment.
    '''
    neurons = pop_list[:40]
    inputs = pop_list[40:]
    return neurons, inputs

def split_list2(pop_list):
    '''
    Splits the population data list into the portion corresponding to the learning rate and the portion for the number if iterations.
    '''
    learning_rate = pop_list[:10]
    iterations = pop_list[10:]
    return learning_rate,iterations

def count_chunk(chunk):
    '''
    Counts the number of ones in a chunk.
    '''
    count = 0
    for x in range(np.shape(chunk)[0]):
        if chunk[x] == 1:
            count += 1
    return count

def get_input_adjust(chunk):
    '''
    Converts the sequence into a numpy array, a value of 0 will deactivate a weight, while 1 will activate it. 
    '''
    array = np.array(chunk)
    return array
    
def create_sets():
    '''
    Creates the required sets for running the MLP, includes noramising the data.
    '''
    x = preprocess.preprocess('Pollens')
    pollen = np.array(x.create_one_file(SIMPLE_GRASS))
    pollen = x.normalise_max(pollen)
    train_set, train_set_target, test_set, test_set_target, validation_set, validation_set_target = x.make_groups(pollen, LABEL_SIZE, train_size=450, test_size=0, validation_size=200)    
    return train_set, train_set_target, validation_set, validation_set_target

def run_mlp(train_set, train_set_target, number_neurons, validation_set, validation_set_target, eta, iterations, input_adjust):    
    '''
    Runs the MLP, the fitness score is determined using the error, which is subtracted from 100 to ensure the lowest error has the best fitness.
    '''
    p = ga_mlp.mlp(train_set, train_set_target, number_neurons, input_adjust, outtype = 'softmax')   
    error = p.earlystopping(train_set, train_set_target, validation_set, validation_set_target, eta, iterations)
    if math.isnan(error):
        error = 100
    score = adjust_error(error)   
    print score
    return score

def adjust_error(error):
    error = 100 - error
    return error 
    
def print_response_mlp1(fitness, pop):
    '''
    Prints information on the fittest population member.
    '''  
    max_index, score = get_best(fitness)
    neurons_chunk, input_chunk = split_list1(pop[max_index]) 
    number_neurons = count_chunk(neurons_chunk)
    on_inputs, off_inputs = get_on_off_inputs(input_chunk)
    do_the_print(number_neurons, on_inputs, off_inputs, score)
        
def get_best(fitness):
    '''
    iterates over the fitness array and returns the index of the highest value
    '''
    best = 0
    for x in range(len(fitness)):
        if fitness[x] > fitness[best]:
            best = x   
    return best, fitness[best]
      
def get_on_off_inputs(input_chunk):
    '''
    Iterates over the binary corresponding to the weights and returns the indicies of on and off weights.
    '''
    on_inputs = []
    off_inputs = []
    for x in range(len(input_chunk)):
        if input_chunk[x] == 1:
            on_inputs.append(x)
        else:
            off_inputs.append(x)
    return on_inputs, off_inputs

def do_the_print(number_neurons, on_weights, off_weights, score):  
    print 'The best fitness was %i with %i neurons' % (score, number_neurons)
    print 'The following inputs were on:'
    print on_weights
    print 'The following inputs were off:'
    print off_weights

def print_response_mlp2(fitness, pop):
    '''
    Prints information on the fittest population member.
    '''  
    max_index, score = get_best(fitness)
    learning_rate, iterations = split_list2(pop[max_index]) 
    learning_rate = count_chunk(learning_rate) * 0.05 + 0.05
    iterations = count_chunk(iterations) * 20 + 100
    print "The best fitness was %i with %f learning rate and %i iterations." % (score, learning_rate, iterations) 
    
 
    

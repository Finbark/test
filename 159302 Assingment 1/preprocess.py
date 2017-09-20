'''
Created on 19/08/2017

@author: Finbar Kiddle
'''
import os
import numpy as np
import pylab as pl
from scipy import linalg as la

class preprocess:
    '''
    classdocs
    '''

    def __init__(self, source_folder):
        '''
        Constructor
        '''
        self.new_data = []
        self.source_folder = source_folder
    
    
    def create_one_file(self, simple_grass = False):
        '''
        Iterate over all pollen data files in the source folder folder to create one shuffled data file.
        '''
        for filename in os.listdir(self.source_folder):
            self.add_data(filename, simple_grass)      
        self.new_data = np.array(self.new_data)
        np.random.shuffle(self.new_data)
        return self.new_data
    
    def add_data(self, filename, simple_grass):
        '''
        Combines the individual pollen.dat files into one, includes an additional column for the target value.   
        '''
        path = self.source_folder + '/' + filename
        target = self.get_number(filename)
        f = open(path) 
        for row in f:
            self.manage_simple_grass(row, target, simple_grass)
        f.close()
    
    def get_number(self, filename):
        '''
        Extracts the target value from the filename
        '''
        number = ''
        for c in filename:
            if c.isdigit():
                number += c
        return int(number)     

    def manage_simple_grass(self, row, target, simple_grass):
        '''
        If using simple grass, skip pollen files 9 and 13 and adjust the target name for those files after 9 accordingly.
        '''
        if simple_grass:
            if target == 9 or target == 13:
                pass
            elif target > 9:
                    target -= 1
                    self.append_row(row, target)
            else:
                self.append_row(row, target)
        else:
            self.append_row(row, target)
    
    def append_row(self, row, target):
        '''
        Creates a new row of data, with the target as the last element, and adds it to the master array. 
        '''
        new_row = self.get_row(row)
        new_row.append(target)
        self.new_data.append(new_row)
                        
    def get_row(self, row):
        '''
        Converts the row into an 1 dimensional array of floats.
        '''
        data = row.split()
        data = self.convert_to_float(data)
        return data
    
    def convert_to_float(self, row):
        '''
        Iterates over a row and converts all elements to floats. 
        '''
        for i in range(len(row)):
            row[i] = float(row[i])
        return row
    
    def normalise_max(self, data): 
        '''
        Normalisation using max and min.
        '''
        data[:,0:-1] = data[:,0:-1]-data[:,0:-1].mean(axis=0)
        dmax = np.concatenate((data.max(axis=0)*np.ones((1,44)),data.min(axis=0)*np.ones((1,44))),axis=0).max(axis=0)
        data[:,0:-1] = data[:,0:-1]/dmax[0:-1]
        return data
    
    def make_groups(self, data, label_number, algorithm='mlp', train_size=200, test_size=100, validation_size=100):
        '''
        Splits the data into sets. 
        If the groups are being used for a multi-layered perceptron, process the targets into an 2d array with a one in the column corresponding to the target.
        If the groups are being used for a self-organisng map, leave the targets in a 1d array of integers. 
        '''
        if algorithm == 'mlp':
        
            master_train, master_target = self.split_data(data)             
            train_set = np.array(master_train[:train_size,:])
            train_set_target = master_target[:train_size] 
            train_set_target = self.make_target_set(train_set_target, label_number)  
            test_set = np.array(master_train[train_size:train_size + test_size,:])
            test_set_target = master_target[train_size:train_size + test_size]
            test_set_target = self.make_target_set(test_set_target, label_number)  
            validation_set = np.array(master_train[train_size + test_size: train_size + test_size + validation_size,:])
            validation_set_target = master_target[train_size + test_size: train_size + test_size + validation_size]
            validation_set_target = self.make_target_set(validation_set_target, label_number) 
               
            return train_set, train_set_target, test_set, test_set_target, validation_set, validation_set_target
            
        else:
            master_train, master_target = self.split_data(data)             
            train_set = np.array(master_train[:train_size,:])
            train_set_target = master_target[:train_size] - 1
            test_set = np.array(master_train[train_size:train_size + test_size,:])
            test_set_target = master_target[train_size:train_size + test_size] - 1
            validation_set = np.array(master_train[train_size + test_size: train_size + test_size + validation_size])
            validation_set_target = master_target[train_size + test_size: train_size + test_size + validation_size]
               
            return train_set, train_set_target, test_set, test_set_target, validation_set, validation_set_target
                      
    def split_data(self, data):
        '''
        Splits the targets from the training data.
        '''
        train = data[:,0:-1]
        target = data[:,-1]
        return train, target
    
    def make_target_set(self, data, label_number):
        '''
        Converts the targets from an integer to an array where the index of the one is the target value.
        '''
        size = np.shape(data)[0]
        
        targets = np.zeros((size, label_number))
        for i in range(size):
            value = int(data[i]) - 1
            targets[i, value] = 1 
        return targets    
            

    
        

           
        


# -*- coding: utf-8 -*-

"""
Created on Thu May  3 23:54:20 2018

Mauricio de Oliveira
UFMG - Belo Horizonte, Brazil
oliveiramauricio@dcc.ufmg.br

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from utils import Y_dict, params
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from timeit import default_timer as timer

data = genfromtxt('data_tp1', delimiter=',')

#plot a few inputs
X = data[:,1:]/255 # X 
Y = data[:,0]      # Y

X_train = X
Y_train = np.array([Y_dict[y] for y in Y])


X_train, Y_train, Y = shuffle(X_train, Y_train, Y)

running_times         = {}
trained_networks = {}


for param in params:
    
    start = timer()
    ann = MLPClassifier(**params[param]).fit(X_train, Y_train)
    end = timer()
    
    running_times[param] = end - start
    trained_networks[param] = ann
    
    print(param + " finished.")
    print(running_times[param])
    print(ann.score(X_train, Y_train))


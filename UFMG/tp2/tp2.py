#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 01:45:21 2018

@author: maurice
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from timeit import default_timer as timer

class BestDecisionStump():
    
    """ Fit data to best decision stumps considering x, y and w."""
    
    def __init__(self, x, y, w, 
                 num_cols = 9, split_values = [-0.5, 0.5]):
        
        """Initializes the decision stump ."""
    
        self.x = x
        self.y = y
        self.w = w
        self.split_values = split_values
        self.num_cols = num_cols
        
    def fit(self):
        
        " Returns the best stump available, considering x, y and w. "
        
        stump = np.zeros((self.num_cols * 2, 1))
        
        for num_col in range(self.num_cols):
             
             w_label = self.w*self.y
             stump[num_col] =  w_label[self.x[:,num_col] > self.split_values[0]].sum() 
             stump[num_col] -= w_label[self.x[:,num_col] < self.split_values[0]].sum() 
             stump[self.num_cols + num_col] =  w_label[self.x[:,num_col] > self.split_values[1]].sum() 
             stump[self.num_cols + num_col] -= w_label[self.x[:,num_col] < self.split_values[1]].sum() 
                     
        self.best_stump = {"stump_perf": stump.max(), 
                           "col": np.argmax(stump),
                           "split": self.split_values[0] 
                                    if np.argmax(stump) <= self.num_cols-1 
                                    else self.split_values[1]}
        
        return self.best_stump
    
    def predict(self):
        
        "Must be called after fit."
        
        self.pred = np.zeros(self.y.shape[0])
        
        i = 0
        
        for x in self.x[:,self.best_stump["col"]%9]:
            
            if float(x) > self.best_stump["split"]:
                self.pred[i] = 1.0
            else:
                self.pred[i] = -1.0
            
            i+=1
            
        assert(self.pred.shape == self.y.shape)
        return self.pred
    
    def calculate_error(self):
        
        "Must be called after fit and predit."
        
        return self.w.dot(self.pred != self.y)
    
def make_prediction(stump, x):
    
    pred = np.zeros(x.shape[0])
        
    i = 0
        
    for value in x[:,stump["col"]%9]:
    
        if float(value) > stump["split"]:
            pred[i] = 1.0
        else:
            pred[i] = -1.0
            
        i+=1
        
    return pred

#plt.plot(X_train[:30,4], range(30),"bo")
    
data = genfromtxt('tic-tac-toe', delimiter=',', dtype="|U5")

X = data[:,:-1]
Y = data[:,-1]

X[X == 'x'] = 1.0
X[X == 'o'] = -1.0
X[X == 'b'] = 0.0
X = X.astype(float)

Y = np.where(Y == 'negat', -1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size = 0.3)
                                                    #random_state = 30)
    
#total_iter = 300

def get_scores(total_iter):
    
    """ Returns train, test, validation and average 
        stump error after total_iter iterations. """

    hypotheses   = dict()
    stumps_error = dict()

    ensemble_error = 0
    validation_error = 0
    training_error = 0
    stump_overall_error = 0

    kf = KFold(n_splits = 5)
    k = 0

    for train_index, val_index in kf.split(X_train):      
    
        N = len(train_index)
        w = np.ones(N) / N
    
        for it in range(total_iter):
        
            x_train, x_val = X_train[train_index], X_train[val_index]
            y_train, y_val = Y_train[train_index], Y_train[val_index]

            h = BestDecisionStump(x_train, y_train, w)
            h.fit()
            pred  = h.predict()
            error = h.calculate_error()
            alpha = (np.log((1 - error)/error)) / 2
            stumps_error[it] = {"error":error,"alpha":alpha}

            w = w * np.exp(- alpha * y_train * pred)
            w = w / w.sum()
    
            hypotheses[it] = {"function":h.best_stump, "weight":alpha}

        #average stump error
        stump_error = 0
        alpha_sum = 0
        for it in stumps_error:
            alpha_sum += stumps_error[it]["alpha"]
            stump_error += stumps_error[it]["error"]*stumps_error[it]["alpha"]
        stump_error /= alpha_sum    
        stump_overall_error += stump_error
    
        #training error
        y = np.zeros(x_train.shape[0])
        for it in hypotheses:
            y = y + (hypotheses[it]["weight"] * make_prediction(hypotheses[it]["function"], x_train))
        y = np.sign(y)
        training_error += 1 - accuracy_score(y,y_train)    
    
        #validation error
        y = np.zeros(x_val.shape[0])
        for it in hypotheses:
            y = y + (hypotheses[it]["weight"] * make_prediction(hypotheses[it]["function"], x_val))
        y = np.sign(y)
        validation_error += 1 - accuracy_score(y,y_val)  
    
        #test error
        y = np.zeros(X_test.shape[0])
        for it in hypotheses:
            y = y + (hypotheses[it]["weight"] * make_prediction(hypotheses[it]["function"], X_test))
        y = np.sign(y)
        ensemble_error += 1 - accuracy_score(y,Y_test)  
    
        k+=1
    
    return(training_error/k, validation_error/k, 
           ensemble_error/k, stump_overall_error/k)
    
train_error    = dict()
val_error      = dict()
stump_error    = dict()
test_error     = dict()
    
start = timer()

for n_iter in range(1,500,5):
    
    (e1,e2,e3,e4) = get_scores(n_iter)
    train_error[n_iter]    = e1
    val_error[n_iter]      = e2  
    test_error[n_iter]     = e3  
    stump_error[n_iter]    = e4  

print("total time: ", timer() - start)

fig = plt.figure(figsize=(12,10))

plt.plot(stump_error.keys(), stump_error.values(), color = 'r')
plt.plot(test_error.keys(), test_error.values(), color = 'b')
plt.plot(val_error.keys(), val_error.values(), color = 'g')
plt.plot(train_error.keys(), train_error.values(), color = 'k')

plt.yticks(np.arange(0, 0.5, step=0.05))
plt.legend(["Average Stump Error","Ensemble Test Error",
            "Validation Error", "Training Error"])
plt.xlabel("Número de iterações")
plt.ylabel("Erro")

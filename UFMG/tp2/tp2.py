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
                     
        self.best_stump = {"stump_error": stump.max(), 
                           "col": np.argmax(stump),
                           "split": self.split_values[0] 
                                    if np.argmax(stump) <= 8 
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
                                                    test_size = 0.3, 
                                                    random_state = 30)

N = X_train.shape[0]
"""
X_val_1 = X_train[:int(N/5),:]
X_val_2 = X_train[int(N/5):int(2*N/5),:]
X_val_3 = X_train[int(2*N/5):int(3*N/5),:]
X_val_4 = X_train[int(3*N/5):int(4*N/5),:]
X_val_5 = X_train[int(4*N/5):,:]
X_val = [X_val_1, X_val_2, X_val_3, X_val_4, X_val_5]

Y_val_1 = Y_train[:int(N/5)]
Y_val_2 = Y_train[int(N/5):int(2*N/5)]
Y_val_3 = Y_train[int(2*N/5):int(3*N/5)]
Y_val_4 = Y_train[int(3*N/5):int(4*N/5)]
Y_val_5 = Y_train[int(4*N/5):]

Y_val = [Y_val_1, Y_val_2, Y_val_3, Y_val_4, Y_val_5]
"""
kf = KFold(n_splits = 5)

decision_stumps = []
accuracy = []
    
hypotheses = dict()
w = np.ones(N) / N

ensemble_error = {}
stumps_error   = {}
validation_error = {}

for it in range(10):
                
    h = BestDecisionStump(X_train, Y_train, w)
    h.fit()
    pred  = h.predict()
    error = h.calculate_error()
    #print(error)
    alpha = (np.log((1 - error)/error)) / 2

    w = w * np.exp(- alpha * Y_train * pred)
    w = w / w.sum()
    
    hypotheses[it] = {"function":h.best_stump,"weight":alpha}
    
y = np.zeros(X_train.shape[0])
for it in hypotheses:
    
    y = y + (hypotheses[it]["weight"] * make_prediction(hypotheses[it]["function"], X_train))
    
y = np.sign(y)
print("train error")
print(1 - accuracy_score(y,Y_train))    
 
y = np.zeros(X_test.shape[0])
for it in hypotheses:
    
    y = y + (hypotheses[it]["weight"] * make_prediction(hypotheses[it]["function"], X_test))
    
y = np.sign(y)
print("test error")
print(1 - accuracy_score(y,Y_test))
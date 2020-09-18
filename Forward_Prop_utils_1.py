#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np



def initialize_parameters(layer_dims):
    np.random.seed(3)
    L = len(layer_dims) 
    parameters = {} 
    for l in range(1, L):
        parameters[ "W"+str(l) ] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters[ "b"+str(l) ] = np.zeros((layer_dims[l],1))
    return parameters


def Forward_Prop_Single_Step(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if(activation=='relu'):
#        greaterThanZero = Z > 0  
#        greaterThanZero = greaterThanZero.astype(int) 
#        A = np.multiply(greaterThanZero, Z) 
        leakyZ = np.multiply(0.01, Z)
        A = np.maximum(Z, leakyZ)
    if(activation=='softmax'):
        t = np.exp(Z) 
        sum_t = np.sum(t, axis=0) 
        A = t / sum_t 
    
    linear_cache = (A_prev, W, b)
    activation_cache = (Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache


def L_layer_Forward_Prop(X, Y, parameters):
    L = len(parameters) // 2
    A_prev = X
    caches = [] 
    for l in range(1, L): 
        A, cache = Forward_Prop_Single_Step(A_prev, parameters[ "W"+str(l) ], parameters[ "b"+str(l) ], activation="relu")
        caches.append(cache)
        A_prev = A
        
    AL, cache = Forward_Prop_Single_Step(A_prev, parameters[ "W"+str(L) ], parameters[ "b"+str(L) ], activation="softmax")
    caches.append(cache)
    
    assert(AL.shape == (Y.shape[0], Y.shape[1])) 
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    single_example_cost = - np.sum(np.multiply(Y, np.log(AL)), axis=0)
    cost = np.sum(single_example_cost) / m
    return cost

def accuracy(AL, Y):
    m = Y.shape[1]
    correct = 0
    max_in_each_col = np.amax(AL, axis=0)
    one_hot_AL = np.where(AL<max_in_each_col, 0, 1)
    for i in range(m):
        if (np.array_equal(one_hot_AL[:, i], Y[:, i])):
            correct = correct + 1
    acc = (correct / m) * 100
    return acc

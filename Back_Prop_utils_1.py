
import numpy as np

def back_prop_single_step(dA, Y, cache, activation):
    linear_cache, activation_cache = cache
    Z = activation_cache
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    if(activation=='softmax'):
        #calculate g(Z). (1 -g(Z))
        t = np.exp(Z) #element wise exponent of Z
        sum_t = np.sum(t, axis=0) #for each col. of t, find sum of all rows in that col
        gofZ = t / sum_t #calculate the softmax activation
        #gDashOfZ = np.multiply(gofZ, (1 - gofZ))
        dZ = gofZ - Y
    if(activation=='relu'):
#        temp = Z > 0
#        gDashOfZ = temp.astype(int)
        gDashOfZ = np.ones_like(Z)
        alpha = 0.01
        gDashOfZ[ Z<0 ] = alpha
        dZ = np.multiply(dA, gDashOfZ)
        
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def L_Layer_Back_Prop(AL, Y, caches):
    L = len(caches)
    m = AL.shape[1]
    
    grads = {}
    dAL = - ( np.divide(Y, AL) )
    current_cache = caches[L-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = back_prop_single_step(dAL, Y, current_cache, activation='softmax')
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = back_prop_single_step(grads[ "dA" + str(l + 1) ], Y, current_cache, activation='relu')
        
    return grads
        

def update_parameters(parameters, grads, alpha):
    L = len(parameters) // 2
    for l in range(L):
        parameters[ "W"+str(l+1) ] = parameters[ "W"+str(l+1) ] - ( alpha * grads[ "dW"+str(l+1) ])
        parameters[ "b"+str(l+1) ] = parameters[ "b"+str(l+1) ] - ( alpha * grads[ "db"+str(l+1) ])
    return parameters


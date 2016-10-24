from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
# import your LR training code


# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
### TODO %%%
def pegasos(x, y, l, max_epochs):
    t = 0
    n = len(x)
    lenw= len(x[0]) ##assumes there is at least 1 input
    w = matrix([[0] for i in xrange(lenw)])
    w0=0
    epoch = 0
    while epoch <  max_epochs:
        for i in xrange(n):
            t = t+1
            step = 1./(t*l)
            if (y[i]*(np.dot(w.T,x[i])+w0)<1):
                w = (1-step*l)*w + matrix(step * y[i]*x[i]).T
                w0 += matrix(step * y[i]).T
            else:
                w *= (1 - step * l)
        epoch+=1
    return w0,w


w0,w =  pegasos(X,Y,8,100)

def predict_linearSVM(x):
    return w0 + np.dot(x, w)



# Define the predict_linearSVM(x) function, which uses global trained parameters, w
### TODO: define predict_linearSVM(x) ###

# plot training results
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
pl.show()


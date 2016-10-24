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
epochs = 1000
lmbda = .02
gamma = 2e-2

n= len(X[0])
K = zeros((n,n))
### TODO: Compute the kernel matrix ###
def classifier(X, kernel, ais, x):
    return np.asscalar(ais.T * (matrix([[kernel(x, x2) ] for x2 in X])))


def gaussianKernel(gamma):
    def _gaussianKernel(x1, x2):
        return np.asscalar(np.exp(-1*np.dot((x1-x2).T,(x1-x2))*gamma))
    return _gaussianKernel

def computeKernelMatrix(kernel, X):
    return matrix([[kernel(x1, x2) for x1 in X] for x2 in X])


### TODO: Implement train_gaussianSVM ###
def train_gaussianSVM(x, y, l, kernel, max_epochs):
    kernelMatrix = computeKernelMatrix(kernel, X)
    t = 0
    n = len(x)
    ai = matrix([[0.0] for i in xrange(n)])
    epoch = 0
    while epoch <  max_epochs:
        for i in xrange(n):
            t = t+1
            step = 1./(t*l)
            kernelCol = kernelMatrix[:,i]
            if (y[i]*(np.dot(ai.T,kernelCol))<1):
                ai[i] = (1-step*l)*ai[i] + matrix(step * y[i]).T
            else:
                ai[i] *= (1 - step * l)
        epoch+=1
    return ai


gamma = 4
alpha = train_gaussianSVM(X, Y, lmbda,  gaussianKernel(gamma), epochs)

def predict_gaussianSVM(x):
    return classifier(X, gaussianKernel(gamma), alpha, x)



# Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
### TODO:  define predict_gaussianSVM(x) ###

# plot training results
plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
pl.show()

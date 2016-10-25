from numpy import *
from plotBoundary import *
import pylab as pl
import cvxopt
import numpy as np
# import your SVM training code




####
# We are using the dual with slack. We need equations of the form
## min 1/2x^TPx + q^Tx subject to Gx<= h, Ax =b
# I n our case we have
#max -1/2 sum ai aj yiyj <xi,xj> + sum ai
# subject to sum aiyi =0,  0<ai<C
# and ais are our variables
#therefore, q^T is a vector of -1s
#P is  yiyj <xi,xj>


def linearKernel(x1, x2):
    return np.dot(x1, x2)

def gaussianKernel(gamma):
    def _gaussianKernel(x1, x2):
        return np.asscalar(np.exp(-1*np.dot((x1-x2).T,(x1-x2))*gamma))
    return _gaussianKernel


def classifier(Y, X, kernel, ais, b, x):
    return np.asscalar(matrix(Y).T * diag(ravel(ais)) * (matrix([[kernel(x, x2) ] for x2 in X]))) +b

def getWeights(Y, X, C, kernel):
    lengthAis = len(Y)
    I = diag([1 for i in xrange(lengthAis)])
    q = cvxopt.matrix([-1 for i in xrange(lengthAis)], tc='d')
    P =  cvxopt.matrix(diag(ravel(Y))* (matrix([[kernel(x1,x2) for x1 in X] for x2 in X] )) * diag(ravel(Y)), tc='d')

    ##Have to enforce upper bound and lower bound on ai
    G = cvxopt.matrix(np.concatenate((I,-1*I)),tc='d')
    h = cvxopt.matrix([C for i in xrange(lengthAis)] + [0 for i in xrange(lengthAis)],tc='d')

    A = cvxopt.matrix(Y, tc='d').T
    b = cvxopt.matrix([0], tc='d')
    sol = cvxopt.solvers.qp(P,q,G,h,A,b)

    ais = matrix(sol['x'])

    ##Needs to calculate b. Weight*x +b = 1 for support vectors.
    ##Support vectors have ai between 0 and C
    b=0
    epsilon = 10e-3
    for i in xrange(lengthAis):
        if ais[i]>0+epsilon and ais[i]<C-epsilon:
            b = 1- Y[i]*classifier(Y, X, kernel, ais, 0, X[i])
            break

    nsupportVectors =0
    for i in xrange(lengthAis):
        if ais[i] > 0 + epsilon and ais[i] < C - epsilon:
            nsupportVectors+=1
        if ais[i] < 0 or ais[i] > C :
            print "ERRROR"
            print ais[i]
    print "THERE ARE " + str(nsupportVectors) + " support vectors"
    print sol
    return ais, b


#########
#Question 1:
if False:
    C= 1000
    X = array([[2., 2.], [2., 3.], [0., -1.], [-3., -2.]])
    Y = array([[1.],[1.],[-1.],[-1.]])


    ais, b = getWeights(Y, X, C, linearKernel)

    print ais, b

    def predictSVM(x):
        return classifier(Y, X, linearKernel, ais, b, x)
    plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM practice')

    pl.show()


######question 2 and part of 3:
# parameters
if False:
    name = '2'
    print '======Training======'
    # load data from csv files
    train = loadtxt('data/data'+name+'_train.csv')
    # use deep copy here to make cvxopt happy
    X = train[:, 0:2].copy()
    Y = train[:, 2:3].copy()
    C=0.01

    ais, b = getWeights(Y, X, C, linearKernel)

    def predictSVM(x):
        return classifier(Y, X, linearKernel, ais, b, x)

    errors = 0.
    for i in xrange(len(X)):
        if predictSVM(X[i])*Y[i]<0:
            errors+=1

    print "Classification Error for training is " + str (errors/len(X))

    plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


    print '======Validation======'
    # load data from csv files
    validate = loadtxt('data/data'+name+'_validate.csv')
    Xval = validate[:, 0:2]
    Yval = validate[:, 2:3]

    errors = 0.
    for i in xrange(len(Xval)):
        if predictSVM(Xval[i])*Yval[i]<0:
            errors+=1
    print "Classification Error for validation is " + str (errors/len(Xval))

    # plot validation results
    plotDecisionBoundary(Xval, Yval, predictSVM, [-1, 0, 1], title = 'SVM Validate')
    pl.show()

######question 3, gaussian kernel:
# parameters
if True:
    name = '2'
    print '======Training======'
    # load data from csv files
    train = loadtxt('data/data' + name + '_train.csv')
    # use deep copy here to make cvxopt happy
    X = train[:, 0:2].copy()
    Y = train[:, 2:3].copy()
    C = 0.01
    gamma = 100

    kernel = gaussianKernel(gamma)

    ais, b = getWeights(Y, X, C, kernel)


    def predictSVM(x):
        return classifier(Y, X, kernel, ais, b, x)


    errors = 0.
    for i in xrange(len(X)):
        if predictSVM(X[i]) * Y[i] < 0:
            errors += 1

    print "Classification Error for training is " + str(errors / len(X))

    plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title='SVM Train')

    print '======Validation======'
    # load data from csv files
    validate = loadtxt('data/data' + name + '_validate.csv')
    Xval = validate[:, 0:2]
    Yval = validate[:, 2:3]

    errors = 0.
    for i in xrange(len(Xval)):
        if predictSVM(Xval[i]) * Yval[i] < 0:
            errors += 1
    print "Classification Error for validation is " + str(errors / len(Xval))

    # plot validation results
    plotDecisionBoundary(Xval, Yval, predictSVM, [-1, 0, 1], title='SVM Validate')
    pl.show()
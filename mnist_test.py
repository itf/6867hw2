from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
from sklearn import linear_model
import cvxopt

num_digits = 10
d = 28
num_images = 500
train = 200
val = 150
test = 150
digits = np.empty((num_digits, num_images, d ** 2))
for i in range(num_digits):
    data = loadtxt("data/mnist_digit_" + str(i) + ".csv")
    for j in range(num_images):
        digits[i][j] = 2 * np.array(data[j:j+1, :]) / 255 - 1


""" PART 1 """

# 1 vs 7
train_17 = np.vstack((digits[1][:train], digits[7][:train]))
train_class_17 = np.hstack((np.repeat(1, train), np.repeat(-1, train)))
val_17 = np.vstack((digits[1][train:train+val], digits[7][train:train+val]))
val_class_17 = np.hstack((np.repeat(1, val), np.repeat(-1, val)))
test_17 = np.vstack((digits[1][train+val:], digits[7][train+val:]))
test_class_17 = np.hstack((np.repeat(1, test), np.repeat(-1, test)))

# 3 vs 5
train_35 = np.vstack((digits[3][:train], digits[5][:train]))
train_class_35 = np.hstack((np.repeat(1, train), np.repeat(-1, train)))
val_35 = np.vstack((digits[3][train:train+val], digits[5][train:train+val]))
val_class_35 = np.hstack((np.repeat(1, val), np.repeat(-1, val)))
test_35 = np.vstack((digits[3][train+val:], digits[5][train+val:]))
test_class_35 = np.hstack((np.repeat(1, test), np.repeat(-1, test)))

# 4 vs 9
train_49 = np.vstack((digits[4][:train], digits[9][:train]))
train_class_49 = np.hstack((np.repeat(1, train), np.repeat(-1, train)))
val_49 = np.vstack((digits[4][train:train+val], digits[9][train:train+val]))
val_class_49 = np.hstack((np.repeat(1, val), np.repeat(-1, val)))
test_49 = np.vstack((digits[4][train+val:], digits[9][train+val:]))
test_class_49 = np.hstack((np.repeat(1, test), np.repeat(-1, test)))

# even vs odd
train_all = np.vstack((digits[0][:train],
    digits[2][:train],
    digits[4][:train],
    digits[6][:train],
    digits[8][:train],
    digits[1][:train],
    digits[3][:train],
    digits[5][:train],
    digits[7][:train],
    digits[9][:train]))
train_class_all = np.hstack((np.repeat(1, 5 * train), np.repeat(-1, 5 * train)))
val_all = np.vstack((digits[0][train:train+val],
    digits[2][train:train+val],
    digits[4][train:train+val],
    digits[6][train:train+val],
    digits[8][train:train+val],
    digits[1][train:train+val],
    digits[3][train:train+val],
    digits[5][train:train+val],
    digits[7][train:train+val],
    digits[9][train:train+val]))
val_class_all = np.hstack((np.repeat(1, 5 * val), np.repeat(-1, 5 * val)))
test_all = np.vstack((digits[0][train+val:],
    digits[2][train+val:],
    digits[4][train+val:],
    digits[6][train+val:],
    digits[8][train+val:],
    digits[1][train+val:],
    digits[3][train+val:],
    digits[5][train+val:],
    digits[7][train+val:],
    digits[9][train+val:]))
test_class_all = np.hstack((np.repeat(1, 5 * test), np.repeat(-1, 5 * test)))

""" CHOOSE A MODEL """

test="all"
pegasos = True
linearSVM = True
gaussianSVM = True

print test

if test == "all":
    X = train_all
    Y = train_class_all
    valX = val_all
    valY = val_class_all
    testX = test_all
    testY = test_class_all

if test == "17":
    X = train_17
    Y = train_class_17
    valX = val_17
    valY = val_class_17
    testX = test_17
    testY = test_class_17

if test == "35":
    X = train_35
    Y = train_class_35
    valX = val_35
    valY = val_class_35
    testX = test_35
    testY = test_class_35

if test == "49":
    X = train_49
    Y = train_class_49
    valX = val_49
    valY = val_class_49
    testX = test_49
    testY = test_class_49

model = linear_model.LogisticRegression(penalty='l2', C = 1e0, intercept_scaling=1e2)
model = model.fit(X, Y)
#
# # define the predictLR(x) function, which uses trained parameters
# def predictLR(x):
#     return model.predict_proba(np.array([x]))[0][1]
#
# print('======Training======')
#
# err = 0
# for i in range(len(X)):
#     if Y[i] > 0:
#         if predictLR(X[i]) < 0.5:
#             err += 1
#     else:
#         if predictLR(X[i]) > 0.5:
#             err += 1
#
# print("Training Classification Error Rate: ", err / len(X))
#
# print('======Validation======')
#
# err = 0
# for i in range(len(valX)):
#     if valY[i] > 0:
#         if predictLR(valX[i]) < 0.5:
#             err += 1
#     else:
#         if predictLR(valX[i]) > 0.5:
#             err += 1
#
# print("Validation Classification Error Rate: ", err / len(valX))
#
# print('======Testing======')
#
# err = 0
# for i in range(len(testX)):
#     if testY[i] > 0:
#         if predictLR(testX[i]) < 0.5:
#             err += 1
#     else:
#         if predictLR(testX[i]) > 0.5:
#             err += 1
#
# print("Testing Classification Error Rate: ", err / len(testX))

""" IVAN YOUR CODE """

################################33
if linearSVM:
    C= 0.01
    def linearKernel(x1, x2):
        return np.dot(x1, x2)

    def gaussianKernel(gamma):
        def _gaussianKernel(x1, x2):
            return np.asscalar(np.exp(-1*np.dot((x1-x2).T,(x1-x2))*gamma))
        return _gaussianKernel


    def classifier(Y, X, kernel, ais, b, x):
        return np.asscalar(matrix(Y) * diag(ravel(ais)) * (matrix([[kernel(x, x2) ] for x2 in X]))) +b

    def getWeights(Y, X, C, kernel):
        lengthAis = len(Y)
        I = diag([1 for i in range(lengthAis)])
        q = cvxopt.matrix([-1 for i in range(lengthAis)], tc='d')
        P =  cvxopt.matrix(diag(ravel(Y))* (matrix([[kernel(x1,x2) for x1 in X] for x2 in X] )) * diag(ravel(Y)), tc='d')

        ##Have to enforce upper bound and lower bound on ai
        G = cvxopt.matrix(np.concatenate((I,-1*I)),tc='d')
        h = cvxopt.matrix([C for i in range(lengthAis)] + [0 for i in range(lengthAis)],tc='d')

        A = cvxopt.matrix(Y, tc='d').T
        b = cvxopt.matrix([0], tc='d')
        sol = cvxopt.solvers.qp(P,q,G,h,A,b)

        ais = matrix(sol['x'])

        ##Needs to calculate b. Weight*x +b = 1 for support vectors.
        ##Support vectors have ai between 0 and C
        b=0
        epsilon = 10e-3
        for i in range(lengthAis):
            if ais[i]>0+epsilon and ais[i]<C-epsilon:
                b = 1- Y[i]*classifier(Y, X, kernel, ais, 0, X[i])
                break

        nsupportVectors =0
        for i in range(lengthAis):
            if ais[i] > 0 + epsilon and ais[i] < C - epsilon:
                nsupportVectors+=1
        return ais, b

    print('======Training======')

    ais, b = getWeights(Y, X, C, linearKernel)
    print "C = " +str(C) + " for linear SVM"

    def predictSVM(x):
        return classifier(Y, X, linearKernel, ais, b, x)

    errors = 0.
    for i in range(len(X)):
        if predictSVM(X[i])*Y[i]<0:
            errors+=1

    print("Classification Error for training is " + str (errors/len(X)))

    print('======Validation======')

    def predictSVM(x):
        return classifier(Y, X, linearKernel, ais, b, x)

    errors = 0.
    for i in range(len(valX)):
        if predictSVM(valX[i])*valY[i]<0:
            errors+=1

    print("Classification Error for validation is " + str (errors/len(valX)))

    print('======Testing======')

    def predictSVM(x):
        return classifier(Y, X, linearKernel, ais, b, x)

    errors = 0.
    for i in range(len(testX)):
        if predictSVM(testX[i])*testY[i]<0:
            errors+=1

    print("Classification Error for testing is " + str (errors/len(testX)))

################################
""" PART 2 """

if gaussianSVM:
    print('======Training======')
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

    print "C = " +str(C) + " for gaussian SVM"
    print "gamma = " +str(gamma) + " for gaussian SVM"

    print("Classification Error for training is " + str(errors / len(X)))

    print('======Validation======')

    errors = 0.
    for i in xrange(len(valX)):
        if predictSVM(valX[i]) * valY[i] < 0:
            errors += 1

    print("Classification Error for validation is " + str(errors / len(valX)))

    print('======Testing======')

    errors = 0.
    for i in xrange(len(testX)):
        if predictSVM(testX[i]) * testY[i] < 0:
            errors += 1

    print("Classification Error for testing is " + str(errors / len(testX)))

#####################

""" PART 3 """
if True:
    epochs = 100
    lmbda = .02
    gamma = 100
    print "Lambda = " +str(lmbda) + " for gaussian pegasos"
    print "gamma = " +str(gamma) + " for gaussian pegasos"
    print "epochs = " + str(epochs) + " for gaussian pegasos"

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


    alpha = train_gaussianSVM(X, Y, lmbda,  gaussianKernel(gamma), epochs)

    print("======Training======")
    def predict_gaussianSVM(x):
        return classifier(X, gaussianKernel(gamma), alpha, x)

    errors = 0.
    for i in xrange(len(X)):
        if predict_gaussianSVM(X[i]) * Y[i] < 0:
            errors += 1

    print("Classification Error for training is " + str(errors / len(X)))

    print('======Validation======')

    errors = 0.
    for i in xrange(len(valX)):
        if predict_gaussianSVM(valX[i]) * valY[i] < 0:
            errors += 1

    print("Classification Error for validation is " + str(errors / len(valX)))

    print('======Testing======')

    errors = 0.
    for i in xrange(len(testX)):
        if predict_gaussianSVM(testX[i]) * testY[i] < 0:
            errors += 1

    print("Classification Error for testing is " + str(errors / len(testX)))

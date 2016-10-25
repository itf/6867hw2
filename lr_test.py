from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
import math
from sklearn import linear_model
# import your LR training code

# parameters
name = '1'
print('======Training======')
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = np.array(train[:,0:2])
Y = np.array(train[:,2:3])

def grad_desc(obj_func, grad_func, init_guess, step_size, threshold):
    prev_guess = init_guess
    num_iters = 1
    while True:
        next_guess = prev_guess - step_size * grad_func(prev_guess)
        if linalg.norm(obj_func(next_guess) - obj_func(prev_guess)) < threshold:
            return num_iters, next_guess, obj_func(next_guess)
        prev_guess = next_guess
        num_iters += 1

def grad_func_approx(obj_func, d_step, x):
    grad = np.empty(x.shape)
    for i in range(x.shape[1]):
        d_step_i = np.zeros(x.shape[1])
        d_step_i[i] = d_step / 2
        grad[0][i] = (obj_func(x + d_step_i) - obj_func(x - d_step_i)) / d_step
    return grad

# carry out training
lam = 0

def obj_func_ELR(w_t):
    w_0 = w_t[0][0]
    w = w_t[0][1:]
    w = w[np.newaxis, :]
    return np.log(1 + np.power(math.e, np.multiply(Y.T, (np.dot(w, X.T) + w_0))))[0].sum(axis=0) + lam * linalg.norm(w) ** 2

def grad_func_ELR(w_t):
    return grad_func_approx(obj_func_ELR, 1e-8, w_t)


# _, w_opt, _ = grad_desc(obj_func_ELR, grad_func_ELR, np.array([[0, 0, 0]]), 1e-2, 1e-6)

model = linear_model.LogisticRegression(penalty='l2', C = 1e0, intercept_scaling=1e2)
model = model.fit(X, Y.T[0])

print("Final Weights: ", np.hstack((model.intercept_, model.coef_[0])))

# print("Final Weights: ", w_opt);

# define the predictLR(x) function, which uses trained parameters
def predictLR(x):
    return model.predict_proba(np.array([x]))[0][1]
    # x = np.array([x])
    # w_0 = w_opt[0][0]
    # w = w_opt[0][1:]
    # w = w[np.newaxis, :]
    # return 1 / (1 + math.e ** (np.dot(w, x.T)[0][0] + w_0))

err = 0
for i in range(len(X)):
    if Y[i] > 0:
        if predictLR(X[i]) < 0.5:
            err += 1
    else:
        if predictLR(X[i]) > 0.5:
            err += 1

print("Training Classification Error Rate: ", err / len(X))

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')
#
print('======Validation======')
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

err = 0
for i in range(len(X)):
    if Y[i] > 0:
        if predictLR(X[i]) < 0.5:
            err += 1
    else:
        if predictLR(X[i]) > 0.5:
            err += 1

print("Validation Classification Error Rate: ", err / len(X))

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
pl.show()

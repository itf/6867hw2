from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
from sklearn import linear_model

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

# print('======Training======')
#
# # 1 vs 7
# train_17 = np.vstack((digits[1][:train], digits[7][:train]))
# train_class_17 = np.hstack((np.repeat(1, train), np.repeat(-1, train)))
# test_17 = np.vstack((digits[1][train:train+test], digits[7][train:train+test]))
# test_class_17 = np.hstack((np.repeat(1, test), np.repeat(-1, test)))
# model = linear_model.LogisticRegression(penalty='l2', C = 1e0, intercept_scaling=1e2)
# model = model.fit(train_17, train_class_17)
#
# # define the predictLR(x) function, which uses trained parameters
# def predictLR(x):
#     return model.predict_proba(np.array([x]))[0][1]
#
# err = 0
# for i in range(len(train_17)):
#     if train_class_17[i] > 0:
#         if predictLR(train_17[i]) < 0.5:
#             err += 1
#     else:
#         if predictLR(train_17[i]) > 0.5:
#             err += 1
#
# print("Training Classification Error Rate: ", err / len(train_17))
#
# print('======Testing======')
#
# err = 0
# for i in range(len(test_17)):
#     if test_class_17[i] > 0:
#         if predictLR(test_17[i]) < 0.5:
#             err += 1
#     else:
#         if predictLR(test_17[i]) > 0.5:
#             err += 1
#
# print("Testing Classification Error Rate: ", err / len(test_17))
#
# print('======Training======')
#
# # 3 vs 5
# train_35 = np.vstack((digits[3][:train], digits[5][:train]))
# train_class_35 = np.hstack((np.repeat(1, train), np.repeat(-1, train)))
# test_35 = np.vstack((digits[3][train:train+test], digits[5][train:train+test]))
# test_class_35 = np.hstack((np.repeat(1, test), np.repeat(-1, test)))
# model = linear_model.LogisticRegression(penalty='l2', C = 1e0, intercept_scaling=1e2)
# model = model.fit(train_35, train_class_35)
#
# err = 0
# for i in range(len(train_17)):
#     if train_class_35[i] > 0:
#         if predictLR(train_35[i]) < 0.5:
#             err += 1
#     else:
#         if predictLR(train_35[i]) > 0.5:
#             err += 1
#
# print("Training Classification Error Rate: ", err / len(train_35))
#
# print('======Testing======')
#
# err = 0
# for i in range(len(test_35)):
#     if test_class_35[i] > 0:
#         if predictLR(test_35[i]) < 0.5:
#             err += 1
#     else:
#         if predictLR(test_35[i]) > 0.5:
#             err += 1
#
# print("Testing Classification Error Rate: ", err / len(test_35))
#
# print('======Training======')
#
# # 4 vs 9
# train_49 = np.vstack((digits[4][:train], digits[9][:train]))
# train_class_49 = np.hstack((np.repeat(1, train), np.repeat(-1, train)))
# test_49 = np.vstack((digits[4][train:train+test], digits[9][train:train+test]))
# test_class_49 = np.hstack((np.repeat(1, test), np.repeat(-1, test)))
# model = linear_model.LogisticRegression(penalty='l2', C = 1e0, intercept_scaling=1e2)
# model = model.fit(train_49, train_class_49)
#
# err = 0
# for i in range(len(train_17)):
#     if train_class_49[i] > 0:
#         if predictLR(train_49[i]) < 0.5:
#             err += 1
#     else:
#         if predictLR(train_49[i]) > 0.5:
#             err += 1
#
# print("Training Classification Error Rate: ", err / len(train_49))
#
# print('======Testing======')
#
# err = 0
# for i in range(len(test_49)):
#     if test_class_49[i] > 0:
#         if predictLR(test_49[i]) < 0.5:
#             err += 1
#     else:
#         if predictLR(test_49[i]) > 0.5:
#             err += 1
#
# print("Testing Classification Error Rate: ", err / len(test_49))
#
# print('======Training======')
#
# # even vs odd
# train_all = np.vstack((digits[0][:train],
#     digits[2][:train],
#     digits[4][:train],
#     digits[6][:train],
#     digits[8][:train],
#     digits[1][:train],
#     digits[3][:train],
#     digits[5][:train],
#     digits[7][:train],
#     digits[9][:train]))
# train_class_all = np.hstack((np.repeat(1, 5 * train), np.repeat(-1, 5 * train)))
# test_all = np.vstack((digits[0][train:train+test],
#     digits[2][train:train+test],
#     digits[4][train:train+test],
#     digits[6][train:train+test],
#     digits[8][train:train+test],
#     digits[1][train:train+test],
#     digits[3][train:train+test],
#     digits[5][train:train+test],
#     digits[7][train:train+test],
#     digits[9][train:train+test]))
# test_class_all = np.hstack((np.repeat(1, 5 * test), np.repeat(-1, 5 * test)))
# model = linear_model.LogisticRegression(penalty='l2', C = 1e0, intercept_scaling=1e2)
# model = model.fit(train_all, train_class_all)
#
# err = 0
# for i in range(len(train_all)):
#     if train_class_all[i] > 0:
#         if predictLR(train_all[i]) < 0.5:
#             err += 1
#     else:
#         if predictLR(train_all[i]) > 0.5:
#             err += 1
#
# print("Training Classification Error Rate: ", err / len(train_all))
#
# print('======Testing======')
#
# err = 0
# for i in range(len(test_all)):
#     if test_class_all[i] > 0:
#         if predictLR(test_all[i]) < 0.5:
#             err += 1
#     else:
#         if predictLR(test_all[i]) > 0.5:
#             err += 1
#
# print("Testing Classification Error Rate: ", err / len(test_all))

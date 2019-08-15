#!/usr/bin/env python
# encoding: utf-8
# @Time    : 2019/8/15 12:47 PM
# @Author  : Li Xinjian  
# @File    : two_layer_network.py.py

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from functools import reduce
import operator


X, Y = load_planar_dataset()
print X.shape  # (2, 400)
# print X
print Y.shape  # (1, 400), (0, 1)
# print Y

# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y.reshape(-1,), s=40, cmap=plt.cm.Spectral)
plt.savefig('origin.png')
plt.show()

m = X.shape[1]
print m


clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T.reshape(-1, ))
plot_decision_boundary(lambda x: clf.predict(x), X, Y.reshape(-1, ))
plt.title('Logistic Regression')
plt.savefig('clf_logistic.png')
plt.show()

LR_predictions = clf.predict(X.T)
print 'Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1-Y, 1-LR_predictions))/float(Y.size)*100) + '%'


# 网络的层数，大小
def layer_sizes(X, Y):

    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return n_x, n_h, n_y


# X_assess, Y_assess = layer_sizes_test_case()
# (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
# print("The size of the input layer is: n_x = " + str(n_x))
# print("The size of the hidden layer is: n_h = " + str(n_h))
# print("The size of the output layer is: n_y = " + str(n_y))


# 初始化参数
def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return parameters


# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# 前向计算
def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

    return A2, cache


# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2'])


# 计算损失函数
def compute_cost(A2, Y, parameters):

    m = Y.shape[1]

    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1-Y), np.log((1 - A2)))
    cost = -np.sum(logprobs) / m

    return cost


# A2, Y_assess, parameters = compute_cost_test_case()
# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


# 反向传播
def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache["A1"]
    A2 = cache["A2"]

    # 迭代公式
    dZ2 = A2 - Y
    # print dZ2
    dW2 = (1. / m) * np.dot(dZ2, A1.T)
    db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1. / m) * np.dot(dZ1, X.T)
    db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))


# 开始更新参数
def update_parameters(parameters, grads, learning_rate=1.2):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# 建立模型
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):

    # 网络
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # 初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 参数更新
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print "Cost after iteration %i: %f" % (i, cost)

    return parameters


# X_assess, Y_assess = nn_model_test_case()
# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# 预测
def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions


parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.reshape(-1,))
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.savefig('two_layer_network.png')
plt.show()

predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')












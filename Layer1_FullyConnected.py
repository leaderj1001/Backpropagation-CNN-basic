from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("./sample/MNIST_data/", one_hot=True)
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

def softmax(x):
    x -= np.max(x)
    sm = (np.exp(x).T / np.sum(np.exp(x),axis=1)).T
    return sm

def softmax_deriv(a) :
    return softmax(a) * (1 - softmax(a))

def cross_entropy(X,y,weight):
    m = X.shape[0]
    X_dot = np.dot(X, weight)
    p = softmax(X_dot)
    loss = (-1 / m) * np.sum(np.log(p) * y)
    delta = -np.dot(X.T,(y - p))
    return loss, delta

W = np.random.normal(size=(784, 10))
learning_rate = 0.0001
epochs = 1000

for epoch in range(epochs) :
    batches = int(len(X_train) / 550)

    for i in range(batches):
        mini_X_train = X_train[i * 550 :(i + 1) * 550]
        mini_y_train = y_train[i * 550 :(i + 1) * 550]

        cost, delta = cross_entropy(mini_X_train, mini_y_train,W)
        W -= (learning_rate * delta)

    print('Cost :: %lf' % (cost))
import numpy as np
import mnist

#mnist = input_data.read_data_sets("./sample/MNIST_data/", one_hot=True)
X_train = mnist.train_images()
y_train = mnist.train_labels()
X_test = mnist.test_images()
y_test = mnist.test_labels()


def image_show(img) :

    import matplotlib.pyplot as plt
    plt.imshow(np.squeeze(img))
    plt.show()

# def ReLU(x):
#     return x * (x > 0)
#
# def dReLU(x):
#     return 1. * (x > 0)

def relu(x, Derivative=False):
    if not Derivative:
        return np.maximum(0, x)
    else:
        out = np.ones(x.shape)
        out[(x < 0)] = 0
        return out


X_train = X_train.reshape(-1, 1, 28, 28)
class conv2d():

    def __init__(self, input_channel, output_channel, filter_size, stride):
        # weight filter size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.filter_size = filter_size
        self.stride = stride
        # output_channel의 갯수 만큼 weight 생성
        self.weight = np.random.normal(size=(self.output_channel,self.input_channel, self.filter_size, self.filter_size))

    def forward(self, X):
        # (mini_batch, input_channel, input_size, input_size)
        self.X = X
        self.input_size = self.X.shape[2] # self.X.shape[3]
        self.mini_batch = self.X.shape[0]
        # output에 대한 size
        self.output_size = int((self.input_size - self.filter_size) / self.stride) + 1
        self.output = np.zeros([self.mini_batch, self.output_channel, self.output_size, self.output_size])

        # batch 만큼 먼저 돈다.
        for batch in range(self.mini_batch) :
            # 만들고 싶은 채널의 갯수만큼 돈다.
            for out_channel in range(self.output_channel):
                # input channel의 갯수만큼 계산을 해줘야 하기 때문( ex, 28x28x3 ) 일 때
                for in_channel in range(0, self.input_channel):
                    for i in range(0, self.output_size):
                        for j in range(0, self.output_size):
                            sum = 0
                            for idx1, p in enumerate(range(i * self.stride, i + self.filter_size)):
                                for idx2, q in enumerate(range(j * self.stride, j + self.filter_size)):
                                    sum += (self.X[batch][in_channel][p][q] * self.weight[out_channel][in_channel][idx1][idx2])
                            self.output[batch][out_channel][int(i / self.stride)][int(j / self.stride)] += sum
        return self.output

    def backward(self, learning_rate, beforeDelta):

        for batch in range(self.mini_batch) :
            for out_channel in range(self.output_channel):
                # input channel의 갯수만큼 계산을 해줘야 하기 때문
                for in_channel in range(0, self.input_channel):
                    for i in range(0, self.filter_size):
                        for j in range(0, self.filter_size):
                            nowDelta = 0
                            for idx1, p in enumerate(range(i, i + beforeDelta.shape[2])):
                                for idx2, q in enumerate(range(j, j + beforeDelta.shape[3])):
                                    nowDelta += (self.X[batch][in_channel][p][q] * beforeDelta[batch][out_channel][idx1][idx2])
                            self.weight[out_channel][in_channel][int(i / self.stride)][int(j / self.stride)] -= (learning_rate * nowDelta)

        rotateDelta = np.rot90(np.rot90(beforeDelta))
        newDelta = np.zeros([self.mini_batch, self.input_channel, self.input_size, self.input_size])
        weight_padding = np.pad(self.weight, ((0, 0), (0, 0), (beforeDelta.shape[2] - 1, beforeDelta.shape[3] - 1),
                                 (beforeDelta.shape[2] - 1, beforeDelta.shape[3] - 1)), 'constant')

        for batch in range(self.mini_batch) :
            # current gradient의 갯수는 3개임 weight 갯수가 3개 였기 때문
            for out_channel in range(self.output_channel):
                # 각각의 channel에서 계산한 값 다 더해서 하나의 gradient 만듦.
                for in_channel in range(self.input_channel):
                    for weight_i in range(self.X.shape[2]):  # size = input의 크기
                        for weight_j in range(self.X.shape[3]):
                            for index1, gradient_i in enumerate(range(weight_i, weight_i + rotateDelta.shape[2])):
                                for index2, gradient_j in enumerate(range(weight_j, weight_j + rotateDelta.shape[3])):
                                    newDelta[batch][in_channel][index1][index2] += (
                                            weight_padding[out_channel][in_channel][gradient_i][gradient_j] *
                                            rotateDelta[batch][out_channel][index1][index2])
        return newDelta

class max_pooling():

    def __init__(self, filter_size):
        # (mini_batch, pooling_input_channel, pooling_input_size, pooling_input_size)
        self.filter_size = filter_size

    def forward(self, X):
        self.X = X
        self.mini_batch = self.X.shape[0]
        self.pooling_input_channel = self.X.shape[1]
        self.pooling_input_size = self.X.shape[2]
        self.upsampling = np.zeros([self.mini_batch, self.pooling_input_channel, self.pooling_input_size, self.pooling_input_size])

        pooling_output_size = int(self.pooling_input_size / self.filter_size)
        result = np.zeros([self.mini_batch, self.pooling_input_channel, pooling_output_size, pooling_output_size])
        for batch in range(self.mini_batch) :
            for k in range(self.pooling_input_channel):
                for i in range(0, self.pooling_input_size - 1, self.filter_size):
                    for j in range(0, self.pooling_input_size - 1, self.filter_size):
                        max = -1000000000
                        maxp = 0
                        maxq = 0
                        for p in (range(i, i + self.filter_size)):
                            for q in (range(j, j + self.filter_size)):
                                if max < self.X[batch][k][p][q]:
                                    max = self.X[batch][k][p][q]
                                    maxp = p
                                    maxq = q
                        self.upsampling[batch][k][maxp][maxq] = 1
                        result[batch][k][int(i / self.filter_size)][int(j / self.filter_size)] = max
        return result

    def backward(self, delta):

        for batch in range(self.mini_batch) :
            for k in range(self.upsampling.shape[1]):
                # 총 크기만큼 돈다 4x4 이면 16번 돈다.
                for i in range(0, self.upsampling.shape[2] - 1, self.filter_size):
                    for j in range(0, self.upsampling.shape[3] - 1, self.filter_size):
                        # filter size만큼 for문 돔 (ex filter_size = 2, 4번 돈다 )
                        for p in (range(i, i + self.filter_size)):
                            for q in (range(j, j + self.filter_size)):
                                if self.upsampling[batch][k][p][q] != 0:
                                    self.upsampling[batch][k][p][q] *= (delta[batch][k][int(i / self.filter_size)][int(j / self.filter_size)])
        return self.upsampling


class make_layer():
    def __init__(self, input_channel, output_channel):
        self.weight = np.random.normal(size=(input_channel, output_channel))

    def forward(self, X):
        self.X = X
        XW = np.dot(self.X, self.weight)
        return XW

    def backward(self):
        delta = self.X.T
        return delta


def softmax(z) :

    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting #각 row 마다 max값 구함.
    e_x = np.exp(z - s)  # 각 row의 max값을 빼준다.
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div

class cross_entropy() :

    def __init__(self, X_dot, y):
        self.X_dot = X_dot
        self.y = y

    def forward(self):
        m = self.X_dot.shape[0]
        self.y_predict = softmax(self.X_dot)
        for i in range(10) :
            print('True ',np.argmax(self.y[i]), " predict ", np.argmax(self.y_predict[i]), np.argmax(self.y[i]) == np.argmax(self.y_predict[i]))

        return (-1 / m) * np.sum(np.multiply(np.log(self.y_predict + 0.001e-10), self.y))

    # softmax & cross entropy 미분 값 반환
    def backward(self) :
        delta = -(self.y - self.y_predict)
        return delta



conv1 = conv2d(1, 32, 3, 1)
conv2 = conv2d(32, 64, 4, 1)
layer1 = make_layer(1600, 10)
pooling1 = max_pooling(2)
pooling2 = max_pooling(2)
# learning rate 0.01 하면 weight가 너무 -가 되서 다 죽어버림. -> 코드 잘못 짜서 그럼..
learning_rate = 0.0001
epochs = 5

# Todo
# 코드 주석 달아야하고,
# mini batch 사용해야하고,
# Fully connected 한 층 더 쌓고,
# 선언만 하면 사용할 수 있게 코드 바꾸고,
# padding도 달아줘야함.
# bias 추가
# 시간복잡도 줄이기

batches = int(len(X_train) / 10)

for epoch in range(epochs):
    for i in range(batches):

        mini_X_train = X_train[i * 10:(i + 1) * 10]
        mini_y_train = y_train[i * 10:(i + 1) * 10]

        convolution1 = conv1.forward(mini_X_train)
        relu_function1 = relu(convolution1, Derivative=False)
        pooling1_result1 = pooling1.forward(relu_function1)
        convolution2 = conv2.forward(pooling1_result1)
        relu_function2 = relu(convolution2, Derivative=False)
        pooling1_result2 = pooling2.forward(relu_function2)

        fully = pooling1_result2.reshape(10, -1)
        l1 = layer1.forward(fully)
        c = cross_entropy(l1, mini_y_train)
        cost = c.forward()
        print(i * 10, cost, learning_rate)
        if cost < 2.5 :
            learning_rate = 0.00001
        else :
            learning_rate = 0.0001

        delta = c.backward()
        fullyConnectedLayerDelta = np.dot(layer1.backward(), delta)
        convolutionDelta = np.dot(delta, layer1.weight.T).reshape(10, 64, 5, 5)
        layer1.weight -= (learning_rate * fullyConnectedLayerDelta)

        deltaPooling2 = pooling2.backward(convolutionDelta)
        deltaRelu2 = relu(convolution2, Derivative=True)
        resultDelta2 = deltaPooling2 * deltaRelu2
        conv1_result2 = conv2.backward(learning_rate, resultDelta2)

        deltaPooling1 = pooling1.backward(conv1_result2)
        deltaRelu1 = relu(convolution1, Derivative=True)
        resultDelta1 = deltaPooling1 * deltaRelu1
        conv1.backward(learning_rate, resultDelta1)
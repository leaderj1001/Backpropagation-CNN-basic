from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("./sample/MNIST_data/", one_hot=True)
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

# fully connected layer 생성
class make_layer() :
    def __init__(self, input_channel, output_channel):
        self.weight = np.random.normal(size=(input_channel, output_channel))

    def forward(self, X):
        self.X = X
        return np.dot(self.X, self.weight)

    def backward(self):
        return self.X.T

class cross_entropy():
    def __init__(self, X_dot, y):
        self.X_dot = X_dot
        self.y = y

    def forward(self, type='train'):
        m = self.X_dot.shape[0]
        self.y_predict = softmax(self.X_dot)

        count = 0
        if type == 'predict' :
            # 평가를 위해 사용
            for i in range(10) :
                y_true = np.argmax(self.y[i])
                y_pre = np.argmax(self.y_predict[i])
                if y_true == y_pre :
                    count += 1
                
        # np.log의 값이 0이 되면 -Inf가 되기 때문에 아주 작은 값을 줘서 -Inf 방지
        return ( -1 / m ) * np.sum(np.log(self.y_predict + 0.0000001e-80) * self.y), count

    def backward(self): # softmax & cross entropy 미분 값 반환
        delta = -(self.y - self.y_predict)
        return delta

def softmax(z):
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting #각 row 마다 max값 구함.
    e_x = np.exp(z - s) # 각 row의 max값을 빼준다.
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

learning_rate = 0.00001
epochs = 5
batches = int(len(X_train) / 10)

layers = []
layer1 = make_layer(784,100)
layer2 = make_layer(100,30)
layer3 = make_layer(30,10)
layers.append(layer1)
layers.append(layer2)
layers.append(layer3)

for epoch in range(epochs) :
    for i in range(batches):
        mini_X_train = X_train[i * 10 :(i + 1) * 10]
        mini_y_train = y_train[i * 10 :(i + 1) * 10]

        # feedforward
        XW = mini_X_train
        for j, eachLayer in enumerate(layers) :
            XW = eachLayer.forward(XW)
        c = cross_entropy(XW, mini_y_train)
        cost, cnt2 = c.forward()
        print("index : ", i * 10, "Cost : ",cost)

        # backpropagation
        delta = c.backward() # cross_entropy backpropagation
        deliver_delta = delta
        for j in range(len(layers) - 1, -1, -1) :
            # 현재 weight update 해줄 gradient 구함.
            current_weight_update_delta = np.dot(layers[j].backward(), deliver_delta)
            # 뒤로 넘겨줄 gradient를 구함.
            deliver_delta = np.dot(deliver_delta, layers[j].weight.T)
            # 현재 weight update
            layers[j].weight -= (learning_rate * current_weight_update_delta)

# 평가
batches_test = int(len(X_test) / 10)
test_total = 0
for i in range(batches_test) :
    mini_X_test = X_test[i * 10:(i + 1) * 10]
    mini_y_test = y_test[i * 10:(i + 1) * 10]

    XW = mini_X_test
    for j, eachLayer_test in enumerate(layers):
        XW = eachLayer_test.forward(XW)
    c = cross_entropy(XW, mini_y_test)
    cost, cnt = c.forward('predict')
    test_total += cnt

print('정확도 : ', test_total / len(X_test) * 100, '%')
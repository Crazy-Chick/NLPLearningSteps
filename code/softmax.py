import numpy as np
import pandas as pd

# construct init parameter
hide_size = 100
df = pd.read_excel('C:/Users/SuperHE/Desktop/leaf.xlsx', nrows=340)
data = df.values
W1 = np.zeros(shape=(15,), dtype=float)
for i in range(hide_size):
    W1 = np.vstack((W1, np.random.rand(1, 15)))
W1 = W1 * 0.001
W1[0][0] = 1
for i in range(36):
    if (i == 0):
        W2 = np.random.rand(hide_size + 1)
    else:
        W2 = np.vstack((W2, np.random.rand(hide_size + 1)))
W2 = W2 * 0.001
alpha = 0.001
learn_loop = 303

# training
def BP_Train():
    global hide_size, data, W1, W2, alpha, learn_loop
    for i in range(learn_loop):

        # construct data
        X = np.array(list(data[i][1:])).reshape(15, 1)
        X[0][0] = 1
        Y = int(data[i][0]) - 1
        hide_out = np.dot(W1, X)
        y_out = np.dot(W2, hide_out)


        # softmax
        sum_ex = 0.0
        soft_y = np.exp(y_out)
        for j in range(len(y_out)):
            sum_ex += soft_y[j][0]
        soft_y = soft_y / sum_ex

        # get gradient
        soft_y[Y][0] -= 1
        '''
        mul_y = np.zeros(shape=(36, 36), dtype=float)
        for j in range(36):
            mul_y[j][j] = soft_y[j][0]
        '''
        gradient_W2 = np.dot(soft_y, hide_out.transpose())
        W2_ = W2[0][1:]
        for j in range(1, 36):
            W2_ = np.vstack((W2_, W2[j][1:]))
        gradient_W1 = np.dot(W2_.transpose(), soft_y)
        for j in range(1, hide_size):
            if (hide_out[j][0] <= 0):
                gradient_W1[j - 1][0] = 0
        
        # change
        for j in range(0, hide_size):
            for k in range(15):
                W1[j + 1][k] -= alpha * gradient_W1[j][0] * X[k][0]
        W2 = np.subtract(W2, alpha * gradient_W2)

def Get(X):

    # get output
    hide_out = np.dot(W1, X)
    y_out = np.dot(W2, hide_out)

    # softmax
    sum_ex = 0.0
    soft_y = np.exp(y_out)
    for i in range(len(y_out)):
        sum_ex += soft_y[i][0]
    soft_y = soft_y / sum_ex
    return soft_y

def Test():
    sum_right = 0
    for i in range(304, 340):
        X = np.array(list(data[i][1:])).reshape(15, 1)
        X[0][0] = 1
        Y = int(data[i][0])
        pre_y = Get(X)
        max_ = pre_y[0][0]
        w = 0
        for j in range(1, 36):
            if (max_ < pre_y[j][0]):
                max_ = pre_y[j][0]
                w = j
        w += 1
        if (Y == w):
            sum_right += 1
    print("Accuracy : ", float(sum_right / 37), " right : ", sum_right)

if __name__ == '__main__':
    BP_Train()
    Test()
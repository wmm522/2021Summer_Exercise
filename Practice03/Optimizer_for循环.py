import math
import numpy as np

points = ([[0.0, 0.9],[0.5,2.1],[0.8,2.7],[1.1,3.1],[1.5,4.1],
           [1.9, 4.8],[2.2,5.1],[2.4,5.9],[2.6,6.0],[3.0, 7.0]]) #十个点的坐标数据

num_iterations = 100
N = float(len(points))
#y=w*x+b
#损失函数  f = np.sum[(y - y_hat)^2]/2n
#梯度下降 给定b 和 w 训练数据 学习率

def compute_loss(y, y_hat):
    loss = np.square(y - y_hat) / 2
    return loss

#随机梯度下降
def SGD(points, learningRate):
    new_w = 0
    new_b = 0
    #迭代100次
    for i in range(0, num_iterations):
        loss = 0
        #每个节点进行w更新和b更新，并计算损失值
        for j in range(0,len(points)):
            x = points[j][0]
            y = points[j][1]
            y_hat = new_w * x + new_b
            loss += compute_loss(y, y_hat)
            dw = - x * (y - y_hat)
            db = y_hat - y
            new_w = new_w - learningRate * dw
            new_b = new_b - learningRate * db
        loss /= N
        print('第{}轮迭代，SGD的loss为{}'.format(i+1, loss))


#Momentum梯度下降
def momentum(points, learningRating, beta1):
    new_w = 0
    new_b = 0
    vdw = 0
    vdb = 0
    dw = 0
    db = 0
    for i in range(0, num_iterations):
        loss = 0
        for j in range(0, len(points)):
            x = points[j][0]
            y = points[j][1]
            y_hat = new_w * x + new_b
            # dw,db
            dw += - x * (y - y_hat)
            db += y_hat - y
            loss += compute_loss(y, y_hat)
        dw /= N
        db /= N
        loss /= N
        vdw = beta1 * vdw + (1 - beta1) * dw
        vdb = beta1 * vdb + (1 - beta1) * db
        new_w = new_w - learningRating * vdw
        new_b = new_b - learningRating * vdb
        print('第{}轮迭代，Momentum的loss为{}'.format(i+1, loss))

#RMSprop优化器
def RMSprop(points, learningRating, beta2, a):
    new_w = 0
    new_b = 0
    sdw = 0
    sdb = 0
    dw = 0
    db = 0
    for i in range(0, num_iterations):
        loss = 0
        for j in range(0, len(points)):
            x = points[j][0]
            y = points[j][1]
            y_hat = new_w * x + new_b
            # dw,db
            dw += - x * (y - y_hat)
            db += y_hat - y
            loss += compute_loss(y, y_hat)
        dw /= N
        db /= N
        loss /= N
        sdw = beta2 * sdw + (1 - beta2) * (dw**2)
        sdb = beta2 * sdb + (1 - beta2) * (db**2)
        new_w = new_w - learningRating * dw / np.sqrt(sdw + a)
        new_b = new_b - learningRating * db / np.sqrt(sdb + a)
        print('第{}轮迭代，RMSprop的loss为{}'.format(i+1, loss))

#Adam优化器
def Adam(points, learningRating, beta1, beta2, a):
    new_w = 0
    new_b = 0
    vdw = 0
    vdb = 0
    sdw = 0
    sdb = 0
    dw = 0
    db = 0
    for i in range(1, num_iterations + 1):
        loss = 0
        for j in range(0, len(points)):
            x = points[j][0]
            y = points[j][1]
            y_hat = new_w * x + new_b
            # dw,db
            dw += - x * (y - y_hat)
            db += y_hat - y
            loss += compute_loss(y, y_hat)
        dw /= N
        db /= N
        loss /= N
        vdw = beta1 * vdw + (1 - beta1) * dw
        vdb = beta1 * vdb + (1 - beta1) * db
        sdw = beta2 * sdw + (1 - beta2) * (dw ** 2)
        sdb = beta2 * sdb + (1 - beta2) * (db ** 2)
        vdw_correct = vdw / (1 - beta1 ** i)
        vdb_correct = vdb / (1 - beta1 ** i)
        sdw_correct = sdw / (1 - beta2 ** i)
        sdb_correct = sdb / (1 - beta2 ** i)
        new_w = new_w - learningRating * vdw_correct / np.sqrt(sdw_correct + a)
        new_b = new_b - learningRating * vdb_correct / np.sqrt(sdb_correct + a)
        print('第{}轮迭代，Adam的loss为{}'.format(i, loss))

loss1 = SGD(points, 0.01)
print('================================================')
loss2 = momentum(points, 0.01, 0.9)
print('================================================')
loss3 = RMSprop(points, 0.01, 0.999, 10e-8)
print('================================================')
loss4 = Adam(points, 0.1, 0.9, 0.999, 10e-8)
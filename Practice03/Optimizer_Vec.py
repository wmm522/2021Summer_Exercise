import numpy as np

X = np.array([0.0, 0.5, 0.8, 1.1, 1.5, 1.9, 2.2, 2.4, 2.6, 3.0]).reshape(1,10)
Y = np.array([0.9, 2.1, 2.7, 3.1, 4.1, 4.8, 5.1, 5.9, 6.0, 7.0]).reshape(1,10)
total_loss = 0
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
ε = 10e-8

def initial_para(): # 初始化 维度为1 1
    w = np.zeros((1,1))
    b = np.zeros((1,1))
    dw = np.zeros((1,1))
    db =np.zeros((1,1))
    vdw = np.zeros((1,1))
    vdb = np.zeros((1,1))
    sdw = np.zeros((1,1))
    sdb = np.zeros((1,1))
    grads = {"dw": dw, "db": db}
    params = {"w": w, "b": b, "vdw": vdw, "vdb": vdb, "sdw": sdw, "sdb":sdb}
    return grads, params

def compute_loss(y, y_hat):
    n = y_hat.shape[1] #样本数量
    loss = np.sum(np.square(y - y_hat)) / (2 * n)
    return loss

def forward_prop(X, params):
    w = params["w"]
    b = params["b"]
    y_hat = np.dot(w, X) + b
    return y_hat

def back_prop(X, Y, y_hat, grads):
    n = y_hat.shape[1] #样本数量
    dz = -(Y - y_hat)
    dw = np.dot(dz, X.T) / n
    db = np.sum(dz, axis=1, keepdims=True)
    grads["dw"] = dw
    grads["db"] = db
    return grads

def SGD(learning_rate , grads , params):
    dw = grads["dw"]
    db = grads["db"]
    w = params["w"]
    b = params["b"]

    w = w - learning_rate * dw
    b = b - learning_rate * db

    params["w"] = w
    params["b"] = b
    return params


def Momentum(learning_rate, grads, params, beta1):
    dw = grads["dw"]
    db = grads["db"]
    w = params["w"]
    b = params["b"]
    vdw = params["vdw"]
    vdb = params["vdb"]

    vdw = beta1 * vdw + (1 - beta1) * dw
    vdb = beta1 * vdb + (1 - beta1) * db
    w = w - learning_rate * vdw
    b = b - learning_rate * vdb

    params["vdw"] = vdw
    params["vdb"] = vdb
    params["w"] = w
    params["b"] = b
    return params

def RMSprop(learning_rate, grads, params, beta2, ε):
    dw = grads["dw"]
    db = grads["db"]
    w = params["w"]
    b = params["b"]
    sdw = params["sdw"]
    sdb = params["sdb"]
    sdw = beta2 * sdw + (1 - beta2) * (dw**2)
    sdb = beta2 * sdb + (1 - beta2) * (db**2)
    w = w - learning_rate * dw / (np.sqrt(sdw) + ε)
    b = b - learning_rate * db / (np.sqrt(sdb) + ε)

    params["sdw"] = sdw
    params["sdb"] = sdb
    params["w"] = w
    params["b"] = b
    return params

def Adam(learning_rate, grads, params, beta2, beta1, iteration, ε):
    dw = grads["dw"]
    db = grads["db"]
    w = params["w"]
    b = params["b"]
    vdw = params["vdw"]
    vdb = params["vdb"]
    sdw = params["sdw"]
    sdb = params["sdb"]
    vdw = beta1 * vdw + (1 - beta1) * dw
    vdb = beta1 * vdb + (1 - beta1) * db
    sdw = beta2 * sdw + (1 - beta2) * (dw**2)
    sdb = beta2 * sdb + (1 - beta2) * (db**2)
    vdw_co = vdw / (1 - beta1**iteration) # **也可以
    vdb_co = vdb / (1 - beta1**iteration)
    sdw_co = sdw / (1 - beta2**iteration)
    sdb_co = sdb / (1 - beta2**iteration)
    w = w - learning_rate * vdw_co / (np.sqrt(sdw_co) + ε)
    b = b - learning_rate * vdb_co / (np.sqrt(sdb_co) + ε)
    params["vdw"] = vdw
    params["vdb"] = vdb
    params["sdw"] = sdw
    params["sdb"] = sdb
    params["w"] = w
    params["b"] = b
    return params

grads, params = initial_para()
for i in range(100):
    t = np.random.randint(0, 9)
    y_hat = forward_prop(X[:, t].reshape(1, 1), params)
    loss = compute_loss(Y[:, t].reshape(1, 1), y_hat)
    grads = back_prop(X[:, t].reshape(1, 1), Y[:, t].reshape(1, 1), y_hat, grads)
    params = SGD(learning_rate, grads, params)
    print('第{}轮迭代，SGD的loss为{}'.format(i+1, loss))

for i in range(100):
    y_hat = forward_prop(X, params)
    loss = compute_loss(Y, y_hat)
    grads = back_prop(X, Y, y_hat, grads)
    params = Momentum(learning_rate, grads, params, beta1)
    print('第{}轮迭代，Momentum的loss为{}'.format(i+1, loss))

for i in range(100):
    y_hat = forward_prop(X, params)
    loss = compute_loss(Y, y_hat)
    grads = back_prop(X, Y, y_hat, grads)
    params = RMSprop(learning_rate, grads, params, beta2, ε)
    print('第{}轮迭代，RMSprop的loss为{}'.format(i+1, loss))

for i in range(1,101):
    y_hat = forward_prop(X, params)
    loss = compute_loss(Y, y_hat)
    grads = back_prop(X, Y, y_hat, grads)
    # print(grads)
    params = Adam(learning_rate, grads, params, beta2, beta1, i, ε)
    # print(params)
    print('第{}轮迭代，Adam的loss为{}'.format(i, loss))
import pandas as pd
import numpy as np
df = pd.read_csv("candy-data.csv")
#`Normalized the winpresent to range(0,1) so that it doesnt dominate the gradient descent
df["winpercent"]=(df["winpercent"]-df["winpercent"].min())/(df["winpercent"].max()-df["winpercent"].min())
y_train = df["chocolate"]
x_train = df.drop(columns=['competitorname', 'chocolate'])
#Converting to numpy
x_train=x_train.to_numpy()
y_train=y_train.to_numpy()
m= x_train.shape[0]
n=x_train.shape[1]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost(X, y, w, b):
    z = X @ w + b
    h = sigmoid(z)
    total_cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return total_cost
def compute_gradient(X, y, w, b):
    z = X@w+b
    h = sigmoid(z)
    dw = (1 / m)*(X.T@(h - y))
    db = (1 / m)*np.sum(h - y)
    return dw, db
def gradient_descent(X, y, w, b, epoch, alpha):
    loss_history = []
    for i in range(epoch):
        dw, db = compute_gradient(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        loss = cost(X, y, w, b)
        loss_history.append(loss)
        if i % 100 == 0:
            print(f"Epoch: {i} | Cost: {loss}")
    return w, b, loss_history
w = np.zeros(n)
b = 0
w_final, b_final, loss_history = gradient_descent(x_train, y_train, w, b, 1001, 0.01)
def model(X,w,b):
    z=X@w+b
    y_pred=sigmoid(z)
    y_predd=(y_pred>=0.5).astype(int)
    return y_predd
y_predd=model(x_train,w_final,b_final)
print(f"\nFinal Weights: \n{w_final}")
print(f"Final Bias: {b_final}")
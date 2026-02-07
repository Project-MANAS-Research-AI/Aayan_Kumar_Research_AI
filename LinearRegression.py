import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df=pd.read_csv("weatherHistory.csv")
# Dropping the non-numeric columns
df=df.drop(columns=['Formatted Date','Daily Summary','Summary','Precip Type'])
x_train=df.drop(columns=['Temperature (C)'])
y_train=df["Temperature (C)"]
# Normalised the input
mu=np.mean(x_train, axis=0)
sigma=np.std(x_train,axis=0)
x_train_norm=(x_train-mu)/sigma
x_train_norm["Loud Cover"]=x_train_norm["Loud Cover"].fillna(0)
# Converting to numpy
x_train_norm=x_train_norm.to_numpy()
y_train=y_train.to_numpy()
m=x_train_norm.shape[0]
n=x_train_norm.shape[1]
np.random.seed(1)
# MSE
def cost(X,Y,w,b):
    pred=X@w+b
    costt=(1/(2*m))*np.sum((pred - Y)**2)
    return costt
# Gives direction to model
def compute_gradient(w,b,X,Y):
    pred=X@w+b
    error=pred-Y
    dw=(1/m)*(X.T@error)
    db=(1/m)*(np.sum(error))
    return dw, db
# This function finds the best fit line's w n b and also gives the MSE of all w n b model checks to find the most optimised w n b
def gradient_descent(X,Y,w,b,alpha,epochs):
    cost_history=[]
    print("Starting the Training...")
    for i in range(epochs+1):
        dw,db=compute_gradient(w,b,X,Y)
        w=w-alpha*dw
        b=b-alpha*db
        cos=cost(X,Y,w,b)
        cost_history.append(cos)
        if i%100==0:
            print(f"epoch: {i} | Cost: {cos}")
    print("Training Complete...")
    return w,b,cost_history
w= np.zeros(n)
b=0
w_op,b_op,cost_history=gradient_descent(x_train_norm,y_train,w,b,0.01,2000)
# The Linear Regression model which predicts based on the best w and b
def model(w,b,x_test):
    return x_test@w+b
y_pred=model(w_op,b_op,x_train_norm)
print("Predicted Temperatures: ",y_pred)
print(f"Optimised weights: {w_op}\nOptimised bais: {b_op}\nCost: {cost_history[2000]}")
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred, alpha=0.5, color='blue', label='Data Points')
min_val = min(np.min(y_train), np.min(y_pred))
max_val = max(np.max(y_train), np.max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, label='Perfect Fit Line')
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Actual vs. Predicted Temperature")
plt.legend()
plt.grid(True)
plt.show()
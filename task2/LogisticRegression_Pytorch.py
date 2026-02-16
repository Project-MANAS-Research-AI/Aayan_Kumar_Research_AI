import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv("candy-data.csv")
# Normalize winpercent
df["winpercent"]=(df["winpercent"]-df["winpercent"].min())/(df["winpercent"].max()-df["winpercent"].min())
y = df["chocolate"]
x = df.drop(columns=['competitorname','chocolate'])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
x_train = x_train.to_numpy()
x_test  = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test  = y_test.to_numpy()
X_train = torch.tensor(x_train, dtype=torch.float32)
X_test  = torch.tensor(x_test, dtype=torch.float32)
Y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
Y_test  = torch.tensor(y_test, dtype=torch.float32).view(-1,1)
X_train = X_train.to("cuda")
X_test  = X_test.to("cuda")
Y_train = Y_train.to("cuda")
Y_test  = Y_test.to("cuda")
m,n = X_train.shape
print(X_train.shape)
# Model
model = nn.Linear(n,1).to("cuda")
#BCE Loss Function
loss_fn = nn.BCEWithLogitsLoss()
#Stochastic Gradient Descent optimiser
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 5000
# Training Loop
for epoch in range(epochs+1):
    logits = model(X_train)
    loss = loss_fn(logits, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
# Testing
model.eval()
with torch.inference_mode():
    preds = torch.sigmoid(model(X_test))
    y_test_pred = (preds >= 0.5).int()
y_test_pred_np = y_test_pred.cpu().numpy()
y_test_np = Y_test.cpu().numpy()
#Metric
accuracy = accuracy_score(y_test_np, y_test_pred_np)
print("Accuracy:", accuracy)

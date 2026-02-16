import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("weatherHistory.csv")
X = df.drop(columns=['Formatted Date','Summary','Daily Summary','Loud Cover'])
y = df['Temperature (C)']
X = X.select_dtypes(include=[np.number])
X = X.fillna(X.mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)
y_test  = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)
X_train = X_train.to("cuda")
X_test  = X_test.to("cuda")
y_train = y_train.to("cuda")
y_test  = y_test.to("cuda")
m,n = X_train.shape
print(X_train.shape)
print("n_features: ",n)
#Model
model = nn.Linear(n, 1).to("cuda")
#Loss Function
loss_fn = nn.MSELoss()
# Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 20000

print("Training started...")
#Training Loop
for epoch in range(epochs):
    preds = model(X_train)
    loss = loss_fn(preds, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | MSE Loss: {loss.item():.4f}")

print("Training finished.")
#Testing Model
model.eval()
with torch.inference_mode():
    test_preds = model(X_test)
    testing_loss=loss_fn(test_preds,y_test)

y_pred_np = test_preds.cpu().numpy()
y_test_np = y_test.cpu().numpy()
#Metrics
mse = mean_squared_error(y_test_np, y_pred_np)
r2 = r2_score(y_test_np, y_pred_np)
print("\n--- Test Performance ---")
print("Loss: ",testing_loss.cpu().numpy())
print("MSE :", mse)
print("R2  :", r2)

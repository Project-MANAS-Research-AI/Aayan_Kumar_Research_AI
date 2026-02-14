import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('pulsar_data_test.csv')
df = df.fillna(df.mean())
x = df.drop(columns='target_class')
#target_class was empty so filled y with 0 and 1 acc to condition(Excess kurtosis of the integrated profile>0.7)
y = (x.iloc[:,2] > 0.7).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#Scaling the data cuz SVM is very sensitive to distance between two data points 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("Linear kernel: ")
svm_linear = SVC(kernel='linear')
svm_linear.fit(x_train, y_train)
print("Linear Accuracy:", svm_linear.score(x_test, y_test))
print("Radial Bias Function kernel: ")
svm_rbf = SVC(kernel='rbf', gamma='scale')
svm_rbf.fit(x_train, y_train)
print("RBF Accuracy:", svm_rbf.score(x_test, y_test))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
df=pd.read_csv("gene_expression.csv")
x=df.drop(columns=['Cancer Present'])
y=df['Cancer Present']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
svm_rbf=SVC(kernel="rbf")
svm_rbf.fit(x_train, y_train)
print("RBF kernel SVM accuracy: ",svm_rbf.score(x_test,y_test))
svm_linear=SVC(kernel="linear")
svm_linear.fit(x_train,y_train)
print("Linear SVM accuracy: ",svm_linear.score(x_test,y_test))




from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve




col_names = ['X', 'Y', 'Z', 'X^2', 'Y^2', 'Z^2', 'X_fd', 'Y_fd', 'Z_fd', 'X_sd', 'Y_sd', 'Z_sd', 'label']
pima=pd.read_csv(<destination_path>,header=None, names=col_names)
pima.head()



feature_cols = ['X', 'Y', 'Z', 'X^2', 'Y^2', 'Z^2', 'X_fd', 'Y_fd', 'Z_fd', 'X_sd', 'Y_sd', 'Z_sd']
X=pima[feature_cols]
Y=pima.label



X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


clf= LogisticRegression(random_state=0)
clf=clf.fit(X_train,Y_train)



accuracy=clf.score(X_test,Y_test)

print(accuracy)

Y_pred=clf.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(Y_test, Y_pred, average='macro')






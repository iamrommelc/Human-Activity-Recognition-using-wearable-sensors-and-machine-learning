
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np



col_names = ['X', 'Y', 'Z', 'X^2', 'Y^2', 'Z^2', 'X_fd', 'Y_fd', 'Z_fd', 'X_sd', 'Y_sd', 'Z_sd', 'label']
pima=pd.read_csv(<dataset_path>,header=None, names=col_names)
pima.head()
feature_cols = ['X', 'Y', 'Z', 'X^2', 'Y^2', 'Z^2', 'X_fd', 'Y_fd', 'Z_fd', 'X_sd', 'Y_sd', 'Z_sd']
X=pima[feature_cols]
Y=pima.label
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)
#KMeans
km = KMeans(n_clusters=7)
km.fit(X_train)
Y_pred=km.predict(X_test)
labels = km.labels_
#Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], X_train.iloc[:, 2],
          c=labels.astype(np.float), edgecolor="k", s=50)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("K Means", fontsize=14)
print(r2_score(Y_test,Y_pred))





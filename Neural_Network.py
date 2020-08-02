
import numpy as np
import pandas as pd
#dataset import
dataset = pd.read_csv(<dataset_path>) 
dataset.head(10)

X = dataset.iloc[:,:12].values
y = dataset.iloc[:,12:13].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


import keras
from keras.models import Sequential
from keras.layers import Dense
# Neural network
model = Sequential()
model.add(Dense(16, input_dim=12, activation="relu"))
model.add(Dense(12, input_dim=12, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(9, activation="softmax"))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



history = model.fit(X_train, y_train, epochs=100, batch_size=64)




y_pred = model.predict(X_test)

pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))




from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8']
print(classification_report(test, pred, target_names=target_names))
print([target_names])
from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)


history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)



import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()


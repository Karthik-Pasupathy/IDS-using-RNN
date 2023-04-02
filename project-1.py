from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (multilabel_confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from datetime import datetime
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from sklearn.utils import resample


start=datetime.now()

traindata = pd.read_csv('#INSERT DATA SET HERE')



#traindata.groupby('attack').counts()

traindata.shape

'''
print('Number of Rows (Samples): %s' % str((traindata.shape[0])))
print('Number of Columns (Features): %s' % str((traindata.shape[1])))
'''
traindata.head(4)
traindata.isnull().sum()
traindata.columns

traindata.info()

traindata['Type'].value_counts()


sns.set(rc={'figure.figsize':(5, 6)})
plt.xlabel('Attack Type')
sns.set_theme()
ax = sns.countplot(x='Type', data=traindata)
ax.set(xlabel='Attack Type', ylabel='Number of Attacks')
plt.show()

sns.set(rc={'figure.figsize':(5, 6)})
sns.scatterplot(x=traindata['Bwd Pkts/s'][:5000000], y=traindata['Fwd Seg Size Min'][:500000],
                hue='Type', data=traindata)
plt.show()


traindata.isna().sum().to_numpy()
cleaned_data = traindata.dropna()
cleaned_data.isna().sum().to_numpy()

label_encoder = LabelEncoder()
cleaned_data['Type']= label_encoder.fit_transform(cleaned_data['Type'])
cleaned_data['Type'].unique()

cleaned_data['Type'].value_counts()

data_1 = cleaned_data[cleaned_data['Type'] == 0]
data_2 = cleaned_data[cleaned_data['Type'] == 1]
data_3 = cleaned_data[cleaned_data['Type'] == 2]


# make benign feature
y_1 = np.zeros(data_1.shape[0])
y_benign = pd.DataFrame(y_1)

# make bruteforce feature
y_2 = np.ones(data_2.shape[0])
y_bf = pd.DataFrame(y_2)

# make bruteforceSSH feature
y_3 = np.full(data_3.shape[0], 2)
y_ssh = pd.DataFrame(y_3)

# merging the original dataframe
X = pd.concat([data_1, data_2, data_3], sort=True)
y = pd.concat([y_benign, y_bf, y_ssh], sort=True)

y_1, y_2, y_3

X.isnull().sum().to_numpy()

data_1_resample = resample(data_1, n_samples=20000,
                           random_state=256, replace=True)
data_2_resample = resample(data_2, n_samples=20000,
                           random_state=256, replace=True)
data_3_resample = resample(data_3, n_samples=20000,
                           random_state=256, replace=True)

train_dataset = pd.concat([data_1_resample, data_2_resample, data_3_resample])
train_dataset.head(2)



test_dataset = train_dataset.sample(frac=0.1)

target_train = train_dataset['Type']
target_test = test_dataset['Type']
target_train.unique(), target_test.unique()


Y = to_categorical(target_train, num_classes=3)
C = to_categorical(target_test, num_classes=3)

train_dataset = train_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Type"], axis=1)
test_dataset = test_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Type"], axis=1)
test_dataset.to_csv('D:/project -1/testdataset.csv')
test_dataset.to_csv('D:/project -1/traindataset.csv')

X = train_dataset.iloc[:, :-1].values
T = test_dataset.iloc[:, :-1].values
#T.to_csv('D:/project -1/T.csv')


scaler = Normalizer().fit(X)
X_train = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])

scaler = Normalizer().fit(T)
X_test = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])
 
y_train = np.array(Y)
y_test = np.array(C)



'''
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
'''

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
X_train.shape, X_test.shape


batch_size = 100

# 1. define the network
model = Sequential()
model.add(SimpleRNN(196,input_dim=72, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(3,activation='sigmoid'))
model.add(SimpleRNN(196, return_sequences=False))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(3,activation='sigmoid'))
#model.layers[3].get_config()
model.summary()

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="D:/project -1/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_accuracy',mode='max')
csv_logger = CSVLogger('training_set_iranalysis1.csv',separator=',', append=False)
#X_train = np.asarray(X_train).astype(np.float32)
#y_train = np.asarray(y_train).astype(np.float32)

model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
model.save('D:/project -1/model/model.hdf5')

print ("\nrun time:" , datetime.now()-start)

model.load_weights('D:/project -1/model/model.hdf5')
print("**********************************************")

loss, accuracy  = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

'''
y_pred = np.argmax(model.predict(X_test),axis =1)
print (y_pred )
# np.save('D:/project -1/predicted.csv', np.transpose([y_test,y_pred])) 
#from sklearn.preprocessing import MultiLabelBinarizer
#recall = recall_score(y_test,y_pred)
precision = precision_score(y_test, y_pred)
#print("\nrecall:")
#print("%.3f" %recall)  
print("\nprecision:")
print("%.3f" %precision)


y_train1 = y_test
y_pred = model.predict(X_test)
print(multilabel_confusion_matrix(y_test, y_pred))
np.savetxt('D:/project -1/predicted.txt', np.transpose([y_test,y_pred]), fmt='%01d') 


recall = recall_score(y_test, y_pred )
precision = precision_score(y_train1, y_pred)
f1 = f1_score(y_train1, y_pred)
print("\nrecall:")
print("%.3f" %recall)
print("\nprecision:")
print("%.3f" %precision)
print("\nf1score:")
print("%.3f" %f1)
#np.savetxt('D:/project -1/predicted.txt', np.transpose([y_test_pred,y_train_pred]),fmt="%s")
#y_pred = model.predict_classes(X_test)
'''
'''
model.load_weights("D:/project -1/model/model.h5")
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
y_train1 = y_test
y_pred = model.predict(X_train)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")
print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("racall")
print("%.3f" %recall)
print("precision")
print("%.3f" %precision)
print("f1score")
print("%.3f" %f1)
 #cm = metrics.confusion_matrix(y_train1, y_pred)
print("==============================================")
'''

ˇ#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:31:18 2019

@author: wuchi
"""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import *
from keras.callbacks import EarlyStopping, History
from sklearn.model_selection import GridSearchCV
from keras.wrappers import scikit_learn
from keras.constraints import max_norm
import h5py
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import load_model

#%%
train = pd.read_csv("train-v3.csv")
validation = pd.read_csv("valid-v3.csv")
test = pd.read_csv("test-v3.csv")
#%%
x_train = train.drop(columns = ['id','price','sale_yr', 'sale_month', 'sale_day'])
y_train = train['price']
x_valid = validation.drop(columns = ['id','price','sale_yr', 'sale_month', 'sale_day'])
y_valid = validation['price']
test = test.drop(columns = ['id','sale_yr', 'sale_month', 'sale_day'])
#%%
X_train = scale(x_train)
X_valid = scale(x_valid)
X_test = scale(test)
#%%
model = Sequential()
model.add(Dense(700, input_dim=X_train.shape[1],  kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(300, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(200, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(18,  kernel_initializer='normal',activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1,  kernel_initializer='normal'))

model.compile(loss='MAE', optimizer= 'adam')

nb_epoch = 500
batch_size = 400

fn=str(nb_epoch)+'_'+str(batch_size)
early_stopping = EarlyStopping(monitor='val_loss', patience=95, verbose=2)

h = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1,validation_data=(X_valid, y_valid),callbacks=[early_stopping])

#%%
model.save(fn+'.h5')
#%%
Y_predict = model.predict(X_test)
np.savetxt('test.csv', Y_predict, delimiter = ',')
#%%

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#%%
print(h.history.keys())     
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right') 
plt.show()
#%%
loss = history.history['loss']
#%%

store = pd.DataFrame(model.predict(X_valid, verbose=1))
store.to_csv('store.csv')
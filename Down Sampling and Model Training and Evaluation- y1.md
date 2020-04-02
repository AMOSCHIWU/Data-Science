##### 載入套件

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import seaborn as sns
```

##### 讀取檔案

```python
train = pd.read_csv('sr_2_4_loan_sr_3_52_feature.csv')

train = train.drop(columns = 'Unnamed: 0')
```

##### 抽取12月以前的資料並將資料分成有購買與沒購買的資料

```python
training_data = train[train['YM'] < 201812]

training_data_0 = training_data[training_data['y1'] == 0]
training_data_1 = training_data[training_data['y1'] == 1]
```

##### 分別從有購買與沒購買的資料抽取10%的檔案當成Validation Data2

```python
validation_2_0 = training_data_0.sample(frac = 0.1)

validation_2_1 = training_data_1.sample(frac = 0.1)

training_data_0 = training_data_0.drop(validation_2_0.index,axis = 0)
training_data_1 = training_data_1.drop(validation_2_1.index,axis = 0)
```

##### 對沒買的資料利用K-means分群並繪製SSE折線圖

```PYTHON
Kmeans0 = training_data_0.iloc[:,2:-1]

Kmeans0 = Kmeans0.values

SSE = []
for k in range(12,26):
    print('Ready...')
    Kmeans_ = KMeans(n_clusters= k, random_state=0).fit(Kmeans0)
    SSE.append(Kmeans_.inertia_)
    print('Finish...')

k_ran = range(12,26)
plt.xlabel('k')

plt.ylabel('SSE')
plt.plot(k_ran,SSE,'o-')
plt.show()
```

##### 從SSE的折線圖後得知檔案分成幾群並取得每列資料所屬的群數

```PYTHON
n_cluster = 39
Kmeans = KMeans(n_clusters= n_cluster, random_state=0).fit(Kmeans0)
cluster = Kmeans.labels_
cluster = pd.DataFrame(cluster)

Kmeans0 = pd.DataFrame(Kmeans0)
Kmeans0['cluster'] = cluster
```

##### 設定迴圈並將每一群的資料分開抽樣後合併

```PYTHON
for iteration in range(1,51):
    frac = 0.02
    sample_0 = Kmeans0[Kmeans0['cluster'] == 0].sample(frac= frac)
    sample_1 = Kmeans0[Kmeans0['cluster'] == 1].sample(frac= frac)
    sample_2 = Kmeans0[Kmeans0['cluster'] == 2].sample(frac= frac)
    sample_3 = Kmeans0[Kmeans0['cluster'] == 3].sample(frac= frac)
    sample_4 = Kmeans0[Kmeans0['cluster'] == 4].sample(frac= frac)
    sample_5 = Kmeans0[Kmeans0['cluster'] == 5].sample(frac= frac)
    sample_6 = Kmeans0[Kmeans0['cluster'] == 6].sample(frac= frac)
    sample_7 = Kmeans0[Kmeans0['cluster'] == 7].sample(frac= frac)
    sample_8 = Kmeans0[Kmeans0['cluster'] == 8].sample(frac= frac)
    sample_9 = Kmeans0[Kmeans0['cluster'] == 9].sample(frac= frac)
    sample_10 = Kmeans0[Kmeans0['cluster'] == 10].sample(frac= frac)
    sample_11 = Kmeans0[Kmeans0['cluster'] == 11].sample(frac= frac)
    sample_12 = Kmeans0[Kmeans0['cluster'] == 12].sample(frac= frac)
    sample_13 = Kmeans0[Kmeans0['cluster'] == 13].sample(frac= frac)
    sample_14 = Kmeans0[Kmeans0['cluster'] == 14].sample(frac= frac)
    sample_15 = Kmeans0[Kmeans0['cluster'] == 15].sample(frac= frac)
    sample_16 = Kmeans0[Kmeans0['cluster'] == 16].sample(frac= frac)
    sample_17 = Kmeans0[Kmeans0['cluster'] == 17].sample(frac= frac)
    sample_18 = Kmeans0[Kmeans0['cluster'] == 18].sample(frac= frac)
    sample_19 = Kmeans0[Kmeans0['cluster'] == 19].sample(frac= frac)
    sample_20 = Kmeans0[Kmeans0['cluster'] == 20].sample(frac= frac)
    sample_21 = Kmeans0[Kmeans0['cluster'] == 21].sample(frac= frac)
    sample_22 = Kmeans0[Kmeans0['cluster'] == 22].sample(frac= frac)
    sample_23 = Kmeans0[Kmeans0['cluster'] == 23].sample(frac= frac)
    sample_24 = Kmeans0[Kmeans0['cluster'] == 24].sample(frac= frac)
    sample_25 = Kmeans0[Kmeans0['cluster'] == 25].sample(frac= frac)
    sample_26 = Kmeans0[Kmeans0['cluster'] == 26].sample(frac= frac)
    sample_27 = Kmeans0[Kmeans0['cluster'] == 27].sample(frac= frac)
    sample_28 = Kmeans0[Kmeans0['cluster'] == 28].sample(frac= frac)
    sample_29 = Kmeans0[Kmeans0['cluster'] == 29].sample(frac= frac)
    sample_30 = Kmeans0[Kmeans0['cluster'] == 30].sample(frac= frac)
    sample_31 = Kmeans0[Kmeans0['cluster'] == 31].sample(frac= frac)
    sample_32 = Kmeans0[Kmeans0['cluster'] == 32].sample(frac= frac)
    sample_33 = Kmeans0[Kmeans0['cluster'] == 33].sample(frac= frac)
    sample_34 = Kmeans0[Kmeans0['cluster'] == 34].sample(frac= frac)
    sample_35 = Kmeans0[Kmeans0['cluster'] == 35].sample(frac= frac)
    sample_36 = Kmeans0[Kmeans0['cluster'] == 36].sample(frac= frac)
    sample_37 = Kmeans0[Kmeans0['cluster'] == 37].sample(frac= frac)
    sample_38 = Kmeans0[Kmeans0['cluster'] == 38].sample(frac= frac)
    sample_39 = Kmeans0[Kmeans0['cluster'] == 39].sample(frac= frac)
    data = pd.concat([sample_0,sample_1,sample_2,sample_3,sample_4,sample_5,sample_6,sample_7,sample_8,sample_9,sample_10,sample_11,sample_12,sample_13,sample_14,sample_15,sample_16,sample_17,sample_18,sample_19,sample_20,sample_21,sample_22,sample_23,sample_24,sample_25,sample_26,sample_27, sample_28,sample_29,sample_30,sample_31,sample_32,sample_33,sample_34,sample_35, sample_36,sample_37,sample_38,sample_39], axis = 0, ignore_index=True)
```

##### 把分群的欄位丟掉

```PYTHON
data = data.drop(columns = 'cluster')
```

##### 		將欄位重新命名並合併有購買的資料


    data.columns = training_data_0.iloc[:,2:-1].columns
    
    data['y1'] = 0
    
    training_data_1_label = training_data_1.iloc[:,2:]
    
    data = pd.concat([data,training_data_1_label],axis = 0)


##### 將資料打亂

```python
data = data.sample(frac = 1)
```

##### 		從裡面抽取10%的資料當成Validation Data1並分割特徵與標籤(剩下為訓練資料)

```PYTHON
validation_1 = data.sample(frac = 0.1)

data = data.drop(validation_1.index,axis = 0)

validation_1_X = validation_1.iloc[:,:-1]
validation_1_Y = validation_1.iloc[:,-1:]
```

##### 將開始切割的Validation Data2 合併並分割特徵與標籤

```PYTHON
validation_2 = pd.concat([validation_2_0,validation_2_1],axis = 0)

validation_2_X = validation_2.iloc[:,2:-1]
validation_2_Y = validation_2.iloc[:,-1:]
```

##### 分割Data的特徵與標籤

```PYTHON
data_X = data.iloc[:,:-1]
data_Y = data.iloc[:,-1:]
```

##### 建立並訓練XGB, RF, GBC, LSTM, GRU模型並取得Validation Data2的F1-SCORE

```PYTHON
'----------------------------------- XG Boost ---------------------------------'
num_round = 20

param = {'booster': 'gbtree', 'objective': 'multi:softmax','num_class': 2, "eval_metric" : "auc", 'max_depth': 2,'eta': 0.5, 'gamma':0.5}

dtrain = xgb.DMatrix(data_X, data_Y)

bst = xgb.train(param,dtrain, num_round)

dvalid1 = xgb.DMatrix(validation_1_X)

validation_1_XGB_pred = bst.predict(dvalid1)

#validation_1_XGB_pred = [round(value) for value in validation_1_XGB_pred]

dvalid2 = xgb.DMatrix(validation_2_X)

validation_2_XGB_pred = bst.predict(dvalid2)

#validation_2_XGB_pred = [round(value) for value in validation_2_XGB_pred]
'-------------------------------- Random Forest -------------------------------'
clf1 = RandomForestClassifier(n_estimators=300, oob_score = True, max_depth = 80)
print('Training RandomForest ... \n')
clf1.fit(data_X,data_Y)
print('Predicting... \n')
validation_1_RF_pred = clf1.predict(validation_1_X)
validation_2_RF_pred = clf1.predict(validation_2_X)
'------------------------------------ GBC ------------------------------------'
clf2 = GradientBoostingClassifier(max_depth =2)
print('Training GBC ... \n')
clf2.fit(data_X,data_Y)
print('Predicting... \n')
validation_1_GBC_pred = clf2.predict(validation_1_X)
validation_2_GBC_pred = clf2.predict(validation_2_X)

'----------------------------------- XG Boost ---------------------------------'
tn, fp, fn, tp = confusion_matrix(validation_2_Y, validation_2_XGB_pred).ravel()
precision = (tp/(tp+fp))
recall = (tp/(tp+fn))
f1_XGB = (2*precision*recall)/(precision+recall)
print('The XGB F1 score is : {} \n'.format(f1_XGB))

'-------------------------------- Random Forest -------------------------------'
tn, fp, fn, tp = confusion_matrix(validation_2_Y, validation_2_RF_pred).ravel()
precision = (tp/(tp+fp))
recall = (tp/(tp+fn))
f1_RF = (2*precision*recall)/(precision+recall)
print('The RF F1 score is : {} \n'.format(f1_RF))

'------------------------------------ GBC ------------------------------------'
#tn, fp, fn, tp = confusion_matrix(validation_1_Y, validation_1_GBC_pred).ravel()
tn, fp, fn, tp = confusion_matrix(validation_2_Y, validation_2_GBC_pred).ravel()
precision = (tp/(tp+fp))
recall = (tp/(tp+fn))
f1_GBC = (2*precision*recall)/(precision+recall)
print('The GBC F1 score is : {} \n'.format(f1_GBC))

'-------------------------  Transform to np.array -----------------------------'
data_X = np.array(data_X)
data_Y = np.array(data_Y)
data_X = np.reshape(data_X, (data_X.shape[0], 1, data_X.shape[1]))

validation_1_X = np.array(validation_1_X)
validation_1_Y = np.array(validation_1_Y)
validation_1_X = np.reshape(validation_1_X, (validation_1_X.shape[0], 1, validation_1_X.shape[1]))
validation_2_X = np.array(validation_2_X)
validation_2_Y = np.array(validation_2_Y)
validation_2_X = np.reshape(validation_2_X, (validation_2_X.shape[0], 1, validation_2_X.shape[1]))
'----------------------------------- LSTM --------------------------------------'
model = Sequential()
model.add(LSTM(300, input_shape = (1, data_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss= 'binary_crossentropy', optimizer="adam" , metrics = ['accuracy'])
model.summary()
nb_epoch = 100
batch_size = 500

fn= 'k_' + str(n_cluster) + '_frac_' + str(frac)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
h = model.fit(data_X, data_Y, epochs= nb_epoch, batch_size= batch_size, validation_data=(validation_1_X, validation_1_Y),callbacks=[early_stopping])

model.save(fn+'.h5')
print(h.history.keys())     
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right') 
plt.show()

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right') 
plt.show()

validation_2_LSTM_pred = model.predict(validation_2_X)
all_f1 = []
criterion = 0.01
prob = []
pp = 0.01
for i in range(1,100):
    validation_2_LSTM_predict = []
    for p in validation_2_LSTM_pred:
        if p >= criterion:
            validation_2_LSTM_predict.append(1)
        else:
            validation_2_LSTM_predict.append(0)
    
    tn, fp, fn, tp = confusion_matrix(validation_2_Y, validation_2_LSTM_predict).ravel()
    precision = (tp/(tp+fp))
    recall = (tp/(tp+fn))
    f1_1 = (2*precision*recall)/(precision+recall)
    #print('The F1 score is : {} \n'.format(f1_1))
    all_f1.append(f1_1)
    criterion += 0.01
    prob.append(pp)
    pp += 0.01


optimal_prob = prob[all_f1.index(max(all_f1))]

validation_2_LSTM_predict = []

for p in validation_2_LSTM_pred:
    if p >= optimal_prob:
        validation_2_LSTM_predict.append(1)
    else:
        validation_2_LSTM_predict.append(0)
            
tn, fp, fn, tp = confusion_matrix(validation_2_Y, validation_2_LSTM_predict).ravel()
precision = (tp/(tp+fp))
recall = (tp/(tp+fn))
f1_LSTM = (2*precision*recall)/(precision+recall)

#print('The LSTM F1 score is : {} \n'.format(f1_LSTM))

'------------------------------------- GRU -----------------------------------'
model1 = Sequential()
model1.add(GRU(300, input_shape = (1, data_X.shape[2])))
model1.add(Dropout(0.2))

model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss= 'binary_crossentropy', optimizer="adam" , metrics = ['accuracy'])
model1.summary()
nb_epoch = 100
batch_size = 800

fn= 'k_' + str(n_cluster) + '_frac_' + str(frac)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
h = model1.fit(data_X, data_Y, epochs= nb_epoch, batch_size= batch_size, validation_data=(validation_1_X, validation_1_Y),callbacks=[early_stopping])

model1.save(fn+'.h5')
#scores = model.evaluate(X_test, y_test, verbose=0)
print(h.history.keys())     
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right') 
plt.show()

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right') 
plt.show()

validation_2_GRU_pred = model1.predict(validation_2_X)

#validation_2_GRU_predict = []
all_f1 = []
criterion = 0.01
prob = []
pp = 0.01
for i in range(1,100):
    validation_2_GRU_predict = []
    for p in validation_2_GRU_pred:
        if p >= criterion:
            validation_2_GRU_predict.append(1)
        else:
            validation_2_GRU_predict.append(0)
    
    tn, fp, fn, tp = confusion_matrix(validation_2_Y, validation_2_GRU_predict).ravel()
    precision = (tp/(tp+fp))
    recall = (tp/(tp+fn))
    f1_1 = (2*precision*recall)/(precision+recall)
    #print('The F1 score is : {} \n'.format(f1_1))
    all_f1.append(f1_1)
    criterion += 0.01
    prob.append(pp)
    pp += 0.01

optimal_prob = prob[all_f1.index(max(all_f1))]

validation_2_GRU_predict = []

for p in validation_2_GRU_pred:
    if p >= optimal_prob:
        validation_2_GRU_predict.append(1)
    else:
        validation_2_GRU_predict.append(0)

tn, fp, fn, tp = confusion_matrix(validation_2_Y, validation_2_GRU_predict).ravel()
precision = (tp/(tp+fp))
recall = (tp/(tp+fn))
f1_GRU = (2*precision*recall)/(precision+recall)

'---------------------------------- Print F1 score ----------------------------'

print('The XGB F1 score is : {} \n'.format(f1_XGB))
print('The RF F1 score is : {} \n'.format(f1_RF))
print('The GBC F1 score is : {} \n'.format(f1_GBC))
print('The LSTM F1 score is : {} \n'.format(f1_LSTM))
print('The GRU F1 score is : {} \n'.format(f1_GRU))
```

##### 建立多數決投票機制(取3個最高的F1 score 做投票)

```python
model_predict = [validation_2_XGB_pred, validation_2_RF_pred, validation_2_GBC_pred, validation_2_LSTM_predict,validation_2_GRU_predict]

model_name = ['XGB','RF','GBC','LSTM','GRU']

each = [f1_XGB,f1_RF,f1_GBC,f1_LSTM,f1_GRU]

f1_best_index = np.argsort(each)

first_model = model_name[f1_best_index[4]]
second_model = model_name[f1_best_index[3]]
third_model = model_name[f1_best_index[2]]

first_f1 = each[f1_best_index[4]]
second_f1 = each[f1_best_index[3]]
third_f1 = each[f1_best_index[2]]

first_pre = model_predict[f1_best_index[4]]
second_pre = model_predict[f1_best_index[3]]
third_pre = model_predict[f1_best_index[2]]

validation_2_voting_result = []
for index in range(len(validation_2_XGB_pred)):
    vote = validation_2_XGB_pred[index] +validation_2_GBC_pred[index]+ validation_2_LSTM_predict+validation_2_GRU_predict
    if vote >= 3:
        validation_2_voting_result.append(1)
    else:
        validation_2_voting_result.append(0)
validation_2_voting_result2=[]
for index in range(len(validation_2_XGB_pred)):
    vote2=validation_2_voting_result[index] +validation_2_RF_pred[index] 
    if vote2 ==0:
        validation_2_voting_result2.append(0)
    else:
        validation_2_voting_result2.append(1)
tn, fp, fn, tp = confusion_matrix(validation_2_Y, validation_2_voting_result2).ravel()
precision = (tp/(tp+fp))
recall = (tp/(tp+fn))
f1_voting_result = (2*precision*recall)/(precision+recall)
    
print('After voting F1 score is : {} \n'.format(f1_voting_result))
```

##### 如果F1-score大於0.07就跳出迴圈

```python
if first_f1 > 0.07 or second_f1 > 0.07 or third_f1 > 0.07 or f1_voting_result > 0.07 :
        break
```
##### 輸出各模型的F1-score長條圖

```python
dictionary = {'Model' : [first_model,second_model,third_model,'Majority_Vote'], 'F1_score':[first_f1,second_f1,third_f1,f1_voting_result]}
bar = pd.DataFrame(dictionary)  
sns.barplot(x="Model", y="F1_score", data= bar).set_title('Loan Best F1 score')
```

##### 預測明年一月會購買信貸的人

```PYTHON
prepare_to_predict = train[train['YM'] == 201812]

prepare_to_predict = prepare_to_predict.drop(columns = ['srno','YM','y1'])

prepare_to_predict = np.array(prepare_to_predict)
prepare_to_predict = np.reshape(prepare_to_predict, (prepare_to_predict.shape[0], 1, prepare_to_predict.shape[1]))

ready_to_predict = model2.predict(prepare_to_predict)

predict_testing_data = []

for p in ready_to_predict:
    if p >= optimal_prob:
        predict_testing_data.append(1)
    else:
        predict_testing_data.append(0)

predict_testing_data = pd.DataFrame(predict_testing_data)

srno = train[train['YM'] == 201812]['srno']

predict_testing_data = predict_testing_data.reset_index()

predict_testing_data = predict_testing_data.drop(columns = 'index')

srno = srno.reset_index()
```

##### 將最後的檔案合併輸出

```PYTHON
srno = srno.drop(columns = 'index')

id_y1 = pd.concat([srno,predict_testing_data],axis = 1,ignore_index = True)

id_y1 = id_y1.rename(columns = {0: 'id',1 : 'y1'})

id_y1.to_csv('GRU_y1_39_0.02_0.076_result.csv',index=False)
```

##### 繪製Confusion Matrix

```python
confusion_matrix = [116823,271,392,25]

confusion_matrix = np.reshape(confusion_matrix,(2,2))

sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, cmap='Blues',fmt='.2%')

plt.figure(figsize=(7,6))
ax = plt.subplot()
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True,ax = ax, cmap='Blues',fmt='.2%', center=0.4)
ax.set_xlabel('Predicted',fontsize=15)
ax.set_ylabel('True',fontsize=15)
ax.set_title('Confusion Matrix',fontsize=20)
```
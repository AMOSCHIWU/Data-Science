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
train = pd.read_csv('sr_2_4_fund_sr_3_46_feature.csv')

train = train.drop(columns = 'Unnamed: 0')
```

##### 抽取12月以前的資料並將資料分成有購買與沒購買的資料

```python
training_data = train[train['YM'] < 201812]

training_data_0 = training_data[training_data['y2'] == 0]
training_data_1 = training_data[training_data['y2'] == 1]
```

##### 分別從有購買與沒購買的資料抽取10%的檔案當成Validation Data2

```python
validation_2_0 = training_data_0.sample(frac = 0.1)

validation_2_1 = training_data_1.sample(frac = 0.1)

training_data_0 = training_data_0.drop(validation_2_0.index,axis = 0)
training_data_1 = training_data_1.drop(validation_2_1.index,axis = 0)
```

##### 對沒買的資料利用K-means分群並繪製SSE折線圖

```python
Kmeans0 = training_data_0.iloc[:,2:-1]

Kmeans0 = Kmeans0.values
SSE = []
for k in range(12,41):
    print('Ready...')
    Kmeans_ = KMeans(n_clusters= k, random_state=0).fit(Kmeans0)
    SSE.append(Kmeans_.inertia_)
    print('Finish...')

k_ran = range(12,41)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(k_ran,SSE,'o-')
plt.show()

```

##### 從SSE的折線圖後得知檔案分成幾群並取得每列資料所屬的群數

```python
n_cluster = 28
Kmeans = KMeans(n_clusters= n_cluster, random_state=0).fit(Kmeans0)
cluster = Kmeans.labels_
cluster = pd.DataFrame(cluster)

Kmeans0 = pd.DataFrame(Kmeans0)
Kmeans0['cluster'] = cluster
```

##### 設定迴圈並將每一群的資料分開抽樣後合併

```python
for iteration in range(1,51):
    frac = 0.06
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
    data = pd.concat([sample_0,sample_1,sample_2,sample_3,sample_4,sample_5,sample_6,sample_7,sample_8,sample_9,sample_10,sample_11,sample_12,sample_13,sample_14,sample_15,sample_16,sample_17,sample_18,sample_19,sample_20,sample_21,sample_22,sample_23,sample_24,sample_25,sample_26,sample_27], axis = 0, ignore_index=True)
```

##### 把分群的欄位丟掉

```python
 data = data.drop(columns = 'cluster')
```

##### 將欄位重新命名並合併有購買的資料

```python
data.columns = training_data_0.iloc[:,2:-1].columns
    
data['y2'] = 0
    
training_data_1_label = training_data_1.iloc[:,2:]
data = pd.concat([data,training_data_1_label],axis = 0)
```

##### 將資料打亂

```python
data = data.sample(frac = 1)
```

##### 從裡面抽取10%的資料當成Validation Data1並分割特徵與標籤(剩下為訓練資料)

```python
validation_1 = data.sample(frac = 0.1)
    
data = data.drop(validation_1.index,axis = 0)
    
validation_1_X = validation_1.iloc[:,:-1]
validation_1_Y = validation_1.iloc[:,-1:]
```

##### 將開始切割的Validation Data2 合併並分割特徵與標籤

```python
validation_2 = pd.concat([validation_2_0,validation_2_1],axis = 0)
    
validation_2_X = validation_2.iloc[:,2:-1]
validation_2_Y = validation_2.iloc[:,-1:]
```

##### 分割Data的特徵與標籤

```python
data_X = data.iloc[:,:-1]
data_Y = data.iloc[:,-1:]
```

##### 建立並訓練XGB, RF, GBC, LSTM, GRU模型並取得Validation Data2的F1-SCORE

```python
'----------------------------------- XG Boost ------------------------------'
    num_round = 20
    
    param = {'booster': 'gbtree', 'objective': 'multi:softmax','num_class': 2, "eval_metric" : "auc", 'max_depth': 6,'eta': 0.2, 'gamma':0}
    
    dtrain = xgb.DMatrix(data_X, data_Y)
    
    bst = xgb.train(param,dtrain, num_round)
    
    dvalid1 = xgb.DMatrix(validation_1_X)
    
    validation_1_XGB_pred = bst.predict(dvalid1)
    
    
    
    dvalid2 = xgb.DMatrix(validation_2_X)
    
    validation_2_XGB_pred = bst.predict(dvalid2)
    
    '-------------------------------- Random Forest -----------------------'
    clf1 = RandomForestClassifier(n_estimators=500, oob_score = True, max_depth = 80)
    print('Training RandomForest ... \n')
    clf1.fit(data_X,data_Y)
    print('Predicting... \n')
    validation_1_RF_pred = clf1.predict(validation_1_X)
    tn, fp, fn, tp = confusion_matrix(validation_1_Y, validation_1_RF_pred).ravel()
    rf_confusion_matrix=confusion_matrix(validation_1_Y, validation_1_RF_pred).ravel()
    precision = (tp/(tp+fp))
    recall = (tp/(tp+fn))
    f1_va1_RF = (2*precision*recall)/(precision+recall)
    print('The RF train F1 score is : {} \n'.format(f1_va1_RF))
    
    validation_2_RF_pred = clf1.predict(validation_2_X)
    tn, fp, fn, tp = confusion_matrix(validation_2_Y, validation_2_RF_pred).ravel()
    rf_confusion_matrix=confusion_matrix(validation_2_Y, validation_2_RF_pred).ravel()
    precision = (tp/(tp+fp))
    recall = (tp/(tp+fn))
    f1_RF = (2*precision*recall)/(precision+recall)
    print('The RF F1 score is : {} \n'.format(f1_RF))
    if  f1_RF > 0.22 :
        break
    
    '------------------------------------ GBC ------------------------------'
    clf2 = GradientBoostingClassifier(max_depth =2)
    print('Training GBC ... \n')
    clf2.fit(data_X,data_Y)
    print('Predicting... \n')
    validation_1_GBC_pred = clf2.predict(validation_1_X)
    validation_2_GBC_pred = clf2.predict(validation_2_X)
    
    '----------------------------------- XG Boost --------------------------'
    
    tn, fp, fn, tp = confusion_matrix(validation_2_Y, validation_2_XGB_pred).ravel()
    precision = (tp/(tp+fp))
    recall = (tp/(tp+fn))
    f1_XGB = (2*precision*recall)/(precision+recall)
    print('The XGB F1 score is : {} \n'.format(f1_XGB))
    
    
    

    
    '------------------------------------ GBC ------------------------------'
    
    
    tn, fp, fn, tp = confusion_matrix(validation_2_Y, validation_2_GBC_pred).ravel()
    precision = (tp/(tp+fp))
    recall = (tp/(tp+fn))
    f1_GBC = (2*precision*recall)/(precision+recall)
    print('The GBC F1 score is : {} \n'.format(f1_GBC))
    
    '-------------------------  Transform to np.array ----------------------'
    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
    data_X = np.reshape(data_X, (data_X.shape[0], 1, data_X.shape[1]))
    
    validation_1_X = np.array(validation_1_X)
    validation_1_Y = np.array(validation_1_Y)
    validation_1_X = np.reshape(validation_1_X, (validation_1_X.shape[0], 1, validation_1_X.shape[1]))
    validation_2_X = np.array(validation_2_X)
    validation_2_Y = np.array(validation_2_Y)
    validation_2_X = np.reshape(validation_2_X, (validation_2_X.shape[0], 1, validation_2_X.shape[1]))
    
    
    '----------------------------------- LSTM ------------------------------'
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
    
    
    '------------------------------------- GRU ------------------------------'
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
    
    '---------------------------------- Print F1 score --------------------'
    
    print('The XGB F1 score is : {} \n'.format(f1_XGB))
    print('The RF F1 score is : {} \n'.format(f1_RF))
    print('The GBC F1 score is : {} \n'.format(f1_GBC))
    print('The LSTM F1 score is : {} \n'.format(f1_LSTM))
    print('The GRU F1 score is : {} \n'.format(f1_GRU))
```

##### 建立多數決投票機制(取4個較低的模型先進行投票再將結果與RF預測結果合併)

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
        vote = validation_2_XGB_pred[index] +validation_2_GBC_pred[index]+ validation_2_LSTM_predict[index]+validation_2_GRU_predict[index]
        if vote >= 3:
            validation_2_voting_result.append(1)
        else:
            validation_2_voting_result.append(0)
    validation_2_voting_result=np.array(validation_2_voting_result)
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

##### 如果F1-score大於0.22就跳出迴圈

```python
 if   f1_RF > 0.22 or   f1_voting_result >0.22:
        break
```

##### 輸出各模型的F1-score長條圖

```PYTHON
dictionary = {'Model' : ['XGB','RF','GBC','GRU','Majority_Vote'], 'F1_score':[f1_XGB,f1_RF,f1_GBC,f1_LSTM,f1_GRU,f1_voting_result]}
bar = pd.DataFrame(dictionary)  
sns.barplot(x="Model", y="F1_score", data= bar).set_title('Fund Best F1 score')
```

##### 預測明年一月會購買基金的人

```python
final_data = train[train['YM'] == 201812]
final_data_X=final_data.iloc[:,2:-1]
final_data_Y=clf1.predict(final_data_X)
```

##### 將最後的檔案合併輸出

```python
srno = train[train['YM'] == 201812]['srno']
srno = srno.reset_index()
srno = srno.drop(columns = 'index')
y2_final_result = pd.concat([srno,final_data_Y],axis = 1,ignore_index = True)
y2_final_result = y2_final_result.rename(columns = {0: 'is',1 : 'y1'})
y2_final_result.to_csv('y2_final_result',index=False)
```

##### 繪製Confusion Matrix圖

```python
confusion_matrix = np.reshape(rf_confusion_matrix,(2,2))

sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, cmap='Blues',fmt='.2%')

plt.figure(figsize=(7,6))
ax = plt.subplot()
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True,ax = ax, cmap='Blues',fmt='.2%', center=0.4)
ax.set_xlabel('Predicted',fontsize=15)
ax.set_ylabel('True',fontsize=15)
ax.set_title('Confusion Matrix',fontsize=20)
```
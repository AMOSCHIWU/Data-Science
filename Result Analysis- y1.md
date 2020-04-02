##### 載入套件

```python
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
```

##### 讀取檔案

```python
train = pd.read_csv('before_encoding_training_data.csv')

train = train.drop(columns = 'Unnamed: 0')

final = pd.read_csv('final_training_data.csv')

final = final.drop(columns = 'Unnamed: 0')

y1_result = pd.read_csv('GRU_y1_39_0.02_0.076_result.csv')
y1_result = y1_result.rename(columns={'id':'srno'})
sr_4_loan = pd.read_csv('sr_4_loan.csv')
sr_4_loan = sr_4_loan.drop(columns = 'Unnamed: 0')
term_and_tf = pd.read_csv('TERM_and_TF.csv')
term_and_tf = term_and_tf.drop(columns = 'Unnamed: 0')
sr_3_TERM = pd.read_csv('sr_3_TERM_is_col.csv')
```

##### 留下12月特徵

```python
train = train.drop(columns = 'y1')
training_data = train[train['YM'] == 201812]
```

##### 將12月的資料與預測1月信貸的結果合併並分出有購買與沒購買的客戶

```python
dec_y1 = pd.merge(training_data,y1_result,on = 'srno', how = 'left')

dec_y1_buy = dec_y1[dec_y1['y1'] == 1]
dec_y1_not_buy = dec_y1[dec_y1['y1'] == 0]
```

##### 比較有購買和沒購買的客戶年齡以及忠誠度

```python
sns.set(style="darkgrid")
buy_n_age = dec_y1_buy['n_age']
age = sns.distplot(buy_n_age,color = 'r', kde=True)

not_buy_n_age = dec_y1_not_buy['n_age']
sns.distplot(not_buy_n_age,color = 'b', kde=True)

buy_scr = dec_y1_buy['scr']
sns.distplot(buy_scr,color = 'r', kde=True)

not_buy_scr = dec_y1_not_buy['scr']
sns.distplot(not_buy_scr,color = 'b', kde=True)
```

##### 將sr_4_loan欄位重新命名

```python
sr_4_loan = sr_4_loan.rename(columns={'level_0':'srno'})

sr_4_loan = sr_4_loan.rename(columns={'level_2':'YM'})

sr_4_loan = sr_4_loan[['srno','YM','level_1','score']]
```

##### 將sr_4_loan合併預測1月信貸的結果並選出12月有購買的客戶資料

```python

sr_4_loan_y1 = pd.merge(sr_4_loan,y1_result,on = 'srno')

sr_4_loan_y1 = sr_4_loan_y1[sr_4_loan_y1['y1'] == 1]

sr_4_loan_y1 = sr_4_loan_y1[sr_4_loan_y1['YM'] == 201812]
```

##### 將欄位轉成數字

```python
labelencoder = LabelEncoder()
sr_4_loan_y1['level_2'] = labelencoder.fit_transform(sr_4_loan_y1['level_1'] )
```

##### 繪製類別與瀏覽量的長條圖

```python
plt.figure(figsize=(15,4))
sns.barplot(x='level_2', y="score", data=sr_4_loan_y1)
```

##### 選出sr_3_Term的12月的資料

```python
sr_3_TERM = sr_3_TERM[sr_3_TERM['DATA_DATE'] == 201812]
```

##### 將欄位名稱進行處理

```python
TERM = sr_3_TERM.iloc[:,2:]
col_rename = []
for i in range(465):
    col_rename.append(i)
col =  term_and_tf.iloc[:,:-1]
col = col.values

columns = []

for name in col:
    columns.append(name[0])

k = TERM[columns]

k = k.fillna(0)

k.columns = col_rename
```

##### 取出sr_3特徵

```python
srno_ym = final.iloc[:,:2]
final = final.drop(columns = 'y1')
sr_3=final.iloc[:,2:]
```

##### 將sr_3合併預測1月信貸的結果並選出12月有購買的客戶資料

```python
srno_ym_sr_3 = pd.concat([srno_ym,sr_3],axis = 1)

srno_ym_sr_3 = srno_ym_sr_3[srno_ym_sr_3['YM'] == 201812]

sr_3 = pd.merge(srno_ym_sr_3, y1_result,on='srno')

sr_3 = sr_3[sr_3['y1'] == 1]
```

##### 將sr_3進行處理

```python
sr_3_T = sr_3.T
sr_3_ = sr_3.iloc[:,2:-1].columns

sr_3_ = sr_3_.values

cc = []
for name in sr_3_:
    cc.append(int(name))

ok = k[cc]

sum_value = ok.sum()

sum_value = pd.DataFrame(sum_value)

sum_value['index1'] = sum_value.index

sum_value = sum_value.rename(columns={'index1':'TERM',0:'Frequency'})
```

##### 繪製字詞與頻率的長條圖

```python
plt.figure(figsize=(18,4))
sns.barplot(x='TERM', y='Frequency' , data=sum_value)
```
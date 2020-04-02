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
y2_result = pd.read_csv('y2_final_result.csv')

train = train.drop(columns = 'Unnamed: 0')
sr_4_fund = pd.read_csv('sr_4_fund.csv')
sr_4_fund = sr_4_fund.drop(columns = 'Unnamed: 0')
final = pd.read_csv('sr_2_4_fund_sr_3_46_feature.csv')

final = final.drop(columns = 'Unnamed: 0')
term_and_tf = pd.read_csv('TERM_and_TF.csv')
term_and_tf = term_and_tf.drop(columns = 'Unnamed: 0')

sr_3_TERM = pd.read_csv('sr_3_group.csv')
```

##### 留下12月特徵

```python

train = train.drop(columns = 'y1')
training_data = train[train['YM'] == 201812]
```

##### 將12月的資料與預測1月基金的結果合併並分出有購買與沒購買的客戶

```python

dec_y2 = pd.merge(training_data,y2_result,on = 'srno', how = 'left')

dec_y2_buy = dec_y2[dec_y2['y2'] == 1]
dec_y2_not_buy = dec_y2[dec_y2['y2'] == 0]
```

##### 比較有購買和沒購買的客戶年齡以及忠誠度

```python
sns.set(style="darkgrid")
buy_n_age = dec_y2_buy['n_age']
age = sns.distplot(buy_n_age,color = 'r', kde=True)

not_buy_n_age = dec_y2_not_buy['n_age']
sns.distplot(not_buy_n_age,color = 'b', kde=True)

buy_scr = dec_y2_buy['scr']
sns.distplot(buy_scr,color = 'r', kde=True)

not_buy_scr = dec_y2_not_buy['scr']
sns.distplot(not_buy_scr,color = 'b',kde=True)
```

##### 將sr_4_fund欄位重新命名

```python
sr_4_fund = sr_4_fund.rename(columns={'level_0':'srno'})

sr_4_fund = sr_4_fund.rename(columns={'level_2':'YM'})

sr_4_fund = sr_4_fund[['srno','YM','level_1','score']]
```

##### 將sr_4_fund合併預測1月基金的結果並選出12月有購買的客戶資料

```python

sr_4_fund_y2 = pd.merge(sr_4_fund,y2_result,on = 'srno')

sr_4_fund_y2 = sr_4_fund_y2[sr_4_fund_y2['y2'] == 1]

sr_4_fund_y2 = sr_4_fund_y2[sr_4_fund_y2['YM'] == 201812]
```

##### 將欄位轉換成數字

```python
labelencoder = LabelEncoder()
sr_4_fund_y2['level_2'] = labelencoder.fit_transform(sr_4_fund_y2['level_1'] )
```

##### 繪製類別與瀏覽量的長條圖

```python
plt.figure(figsize=(15,4))
sns.barplot(x='level_2', y="score", data=sr_4_fund_y2)
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

col = term_and_tf.iloc[:,:-1]
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
final = final.drop(columns = 'y2')
sr_3=final.iloc[:,2:]
```

##### 將sr_3合併預測1月基金的結果並選出12月有購買的客戶資料

```python
srno_ym_sr_3 = pd.concat([srno_ym,sr_3],axis = 1)

srno_ym_sr_3 = srno_ym_sr_3[srno_ym_sr_3['YM'] == 201812]

sr_3 = pd.merge(srno_ym_sr_3, y2_result,on='srno')
sr_3 = sr_3[sr_3['y2'] == 1]
```

##### 將sr_3進行處理

```python
sr_3_T = sr_3.T

sr_3_ = sr_3.iloc[:,300:-1].columns

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
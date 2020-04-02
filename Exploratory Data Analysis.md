##### 載入套件

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
```

##### 讀檔

```python
sr_2 = pd.read_csv('sr_2.csv')
sr_3 = pd.read_csv('sr_3_utf.csv')
sr_4 = pd.read_csv('sr_4_utf.csv')
```

##### 將sr_2金額分成正與負值並繪製產品總類的長條圖

```python
sns.set(style="darkgrid")
sr_2_amt = sns.distplot(sr_2['amt'],color = 'r', kde=True)
sr_2_amt.set_xlim(-200000,200000)
sr_2_amt.set_xlabel("amt", {'size':'14'})
sr_2_amt.set_ylabel("Count", {'size':'14'}) 
sr_2_amt.set_title('sr_2_amt')

product_type = sr_2['prod_type']

sns.set(style="darkgrid")
prod = sns.countplot(x= 'prod_type', data= sr_2)
prod.set_title('sr_2 Product Type')

sr_2_negative = sr_2[sr_2['amt'] < 0]
n = sns.distplot(sr_2_negative['amt'],color = 'b', kde=False)
n.set_xlim(-150000,150000)

sr_2_positive = sr_2[sr_2['amt'] >= 0]
p = sns.distplot(sr_2_positive['amt'],color = 'b', kde=False)
p.set_xlim(-150000,150000)
```

##### 列出sr_3最高頻率的10個TERM

```python
#%%
labelencoder = LabelEncoder()
sr_3['Num_TERM'] = labelencoder.fit_transform(sr_3['TERM'])

re = sns.barplot(x="Num_TERM", y="TF", data=sr_3)

sr_3_group = sr_3.drop(columns = ['srno','DATA_DATE','TERM'])

sr_3_group = sr_3_group.groupby(['Num_TERM']).sum().reset_index()

sr_3_group = sr_3_group.sort_values(by=['TF']).reset_index()

sr_3_group = sr_3_group.iloc[:,2:]

sr_3_10 = sr_3_group.iloc[-10:,:]

ax = sns.barplot(x="Num_TERM", y="TF", data=sr_3_10)
ax.set_title('Most Frequency TERM')

term = sr_3_10['Num_TERM']
term = term.values
term = sorted(term)

chinese = labelencoder.inverse_transform(term)

```

##### 將sr_4瀏覽量最高的10個組合以長條圖呈現

```python
le = LabelEncoder()
sr_4['Num_LV1'] = le.fit_transform(sr_4['TAG_LV1'])
lbe = LabelEncoder()
sr_4['Num_LV2'] = lbe.fit_transform(sr_4['TAG_LV2'])

sr_4_group = sr_4[['Num_LV1','Num_LV2','score','srno','TAG_LV1', 'TAG_LV2', 'UPD_DT']]

sr_4_group = sr_4_group.iloc[:,:3]

sr_4_group = sr_4_group.groupby(['Num_LV1','Num_LV2']).sum().reset_index()

sr_4_group = sr_4_group.sort_values(by=['score']).reset_index()

sr_4_group = sr_4_group.drop(columns = 'index')

sr_4_group_ = sr_4_group.iloc[-10:,:]

level1 = sns.barplot(x ='Num_LV1', y="score", hue = 'Num_LV2',data=sr_4_group_)
level1.set_title('TAG_LV1, TAG_LV1 and Score', {'size':'15'})
level1.set_xlabel("TAG_LV1", {'size':'12'})
level1.set_ylabel("TAG_LV2", {'size':'12'}) 

lv1 = sr_4_group_['Num_LV1']
lv1 = lv1.values
lv2 = sr_4_group_['Num_LV2']
lv2 = lv2.values

chinese_lv1 = le.inverse_transform(lv1)
chinese_lv2 = lbe.inverse_transform(lv2)

corr = sr_4_group.corr()
```


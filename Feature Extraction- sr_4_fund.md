##### 載入套件

```python
import numpy as np
import pandas as pd
from sklearn import preprocessing
```

###### 讀取檔案

```python
sr_4 = pd.read_csv("sr_4_all.csv") 
sr_4 = sr_4.drop("Unnamed: 0", axis = 1)
```

##### 處理日期

```PYTHON
for index, date in enumerate(sr_4['UPD_DT']):
    update = math.floor(date/100)
    sr_4['UPD_DT'][index] = update
    print(index) 
```

##### 查看TAG_LV1欄位

```python
le = preprocessing.LabelEncoder()
le.fit(sr_4['TAG_LV1'])
le.classes_
```

##### 將有關基金的找出來

```python
df1 = sr_4[sr_4['TAG_LV1'].str.contains('保險')]
df2 = sr_4[sr_4['TAG_LV1'].str.contains('地區')]
df3 = sr_4[sr_4['TAG_LV1'].str.contains('展覽')]
df4 = sr_4[sr_4['TAG_LV1'].str.contains('數位金融')]
df5 = sr_4[sr_4['TAG_LV1'].str.contains('旅遊')]
df6 = sr_4[sr_4['TAG_LV1'].str.contains('生涯階段')]
df7 = sr_4[sr_4['TAG_LV1'].str.contains('金融服務')]
df8 = sr_4[sr_4['TAG_LV1'].str.contains('商業經營')]
df9 = sr_4[sr_4['TAG_LV1'].str.contains('外幣')]
df10 = sr_4[sr_4['TAG_LV1'].str.contains('投資理財')]
df11 = sr_4[sr_4['TAG_LV1'].str.contains('新聞雜誌')]
df12 = sr_4[sr_4['TAG_LV1'].str.contains('法律服務')]
df13 = sr_4[sr_4['TAG_LV1'].str.contains('族群')]
df14 = sr_4[sr_4['TAG_LV1'].str.contains('銀行品牌')]

sr_4_y1 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14],axis=0)
```

##### 查看TAG_LV2欄位

```python
le = preprocessing.LabelEncoder()
le.fit(sr_4_y1['TAG_LV2'])
le.classes_
```

##### 將TAG_LV2跟基金沒關係的刪掉

```python
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('上網族')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('健身族')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('台灣中部')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('台灣南部')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('台灣北部')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('台灣東部')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('台灣離島')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('哈日族')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('哈韓族')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('外食族')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('女性媒體')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('御宅族')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('男性媒體')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('華人影視娛樂')]
sr_4_y1 =sr_4_y1[~ sr_4_y1['TAG_LV2'].str.contains('積極型')]
```

##### 依據欄位srno,TAG_LV2,UPD_DT加總 score

```python
UPD_DT = np.array(sr_4_y1['UPD_DT'])
srno = np.array(sr_4_y1['srno'])
TAG_LV2 = np.array(sr_4_y1['TAG_LV2'])   
score = np.array(sr_4_y1['score'])         
sr_4_fund=sr_4_y1['score'].groupby([srno,TAG_LV2,UPD_DT]).sum().to_frame('score').reset_index()
```

##### 輸出檔案

```python
sr_4_fund.to_csv("sr_4_fund.csv")
```


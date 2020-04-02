##### 載入套件

```python
import pandas as pd
import numpy as np
from sklearn import preprocessing
```

##### 讀取檔案

```python
sr_4 = pd.read_csv('sr_4.csv')
sr_4.drop(labels='Unnamed: 0',axis=1)
```

##### 處理日期

```python
for index, date in enumerate(sr_4['UPD_DT']):
    update = math.floor(date/100)
    sr_4['UPD_DT'][index] = update
    print(index) 
```

##### 查看小類

```python
le = preprocessing.LabelEncoder()
le.fit(sr_4['TAG_LV2'])
le.classes_
```

##### 刪掉"TAG_LV2"跟信貸無關的欄位

```python
sr_4_delete1=sr_4[ ~ sr_4['TAG_LV2'].str.contains('7-11')]
sr_4_delete2=sr_4_delete1[ ~ sr_4['TAG_LV2'].str.contains('86小舖')]
sr_4_delete3=sr_4_delete2[ ~ sr_4['TAG_LV2'].str.contains('Android偏好')]
sr_4_delete4=sr_4_delete3[ ~ sr_4['TAG_LV2'].str.contains('Costco')]
sr_4_delete5=sr_4_delete4[ ~ sr_4['TAG_LV2'].str.contains('DIY手作/手工藝')]
sr_4_delete6=sr_4_delete5[ ~ sr_4['TAG_LV2'].str.contains(' IKEA')]
sr_4_delete7=sr_4_delete6[ ~ sr_4['TAG_LV2'].str.contains('LV')]
sr_4_delete8=sr_4_delete7[ ~ sr_4['TAG_LV2'].str.contains('MUJI')]
sr_4_delete9=sr_4_delete8[ ~ sr_4['TAG_LV2'].str.contains('Mac OS偏好')]
sr_4_delete10=sr_4_delete9[ ~ sr_4['TAG_LV2'].str.contains('PANDORA')]
sr_4_delete11=sr_4_delete10[ ~ sr_4['TAG_LV2'].str.contains('PRADA')]
sr_4_delete12=sr_4_delete11[ ~ sr_4['TAG_LV2'].str.contains('Sasa')]
sr_4_delete13=sr_4_delete12[ ~ sr_4['TAG_LV2'].str.contains("Tomod's")]
sr_4_delete14=sr_4_delete13[ ~ sr_4['TAG_LV2'].str.contains('Windows偏好')]
sr_4_delete15=sr_4_delete14[ ~ sr_4['TAG_LV2'].str.contains('ios偏好')]
sr_4_delete16=sr_4_delete15[ ~ sr_4['TAG_LV2'].str.contains('上網族')]
sr_4_delete17=sr_4_delete16[ ~ sr_4['TAG_LV2'].str.contains('京都')]
sr_4_delete18=sr_4_delete17[ ~ sr_4['TAG_LV2'].str.contains('健身族')]
sr_4_delete19=sr_4_delete18[ ~ sr_4['TAG_LV2'].str.contains('全家')]
sr_4_delete20=sr_4_delete19[ ~ sr_4['TAG_LV2'].str.contains('全聯')]
sr_4_delete21=sr_4_delete20[ ~ sr_4['TAG_LV2'].str.contains('冬季')]
sr_4_delete22=sr_4_delete21[ ~ sr_4['TAG_LV2'].str.contains('北海道')]
sr_4_delete23=sr_4_delete22[ ~ sr_4['TAG_LV2'].str.contains('古典音樂')]
sr_4_delete24=sr_4_delete23[ ~ sr_4['TAG_LV2'].str.contains('台式料理')]
sr_4_delete25=sr_4_delete24[ ~ sr_4['TAG_LV2'].str.contains('台灣中部')]
sr_4_delete26=sr_4_delete25[ ~ sr_4['TAG_LV2'].str.contains('台灣北部')]
sr_4_delete27=sr_4_delete26[ ~ sr_4['TAG_LV2'].str.contains('台灣南部')]
sr_4_delete28=sr_4_delete27[ ~ sr_4['TAG_LV2'].str.contains('台灣東部')]
sr_4_delete29=sr_4_delete28[ ~ sr_4['TAG_LV2'].str.contains('台灣離島')]
sr_4_delete30=sr_4_delete29[ ~ sr_4['TAG_LV2'].str.contains('名古屋')]
sr_4_delete31=sr_4_delete30[ ~ sr_4['TAG_LV2'].str.contains('命理')]
sr_4_delete32=sr_4_delete31[ ~ sr_4['TAG_LV2'].str.contains('哈日族')]
sr_4_delete33=sr_4_delete32[ ~ sr_4['TAG_LV2'].str.contains('哈韓族')]
sr_4_delete34=sr_4_delete33[ ~ sr_4['TAG_LV2'].str.contains('商務貿易')]
sr_4_delete35=sr_4_delete34[ ~ sr_4['TAG_LV2'].str.contains('單身(交友)')]
sr_4_delete36=sr_4_delete35[ ~ sr_4['TAG_LV2'].str.contains('國內旅遊')]
sr_4_delete37=sr_4_delete36[ ~ sr_4['TAG_LV2'].str.contains('國際新聞')]
sr_4_delete38=sr_4_delete37[ ~ sr_4['TAG_LV2'].str.contains('地方新聞')]
sr_4_delete39=sr_4_delete38[ ~ sr_4['TAG_LV2'].str.contains('夏季')]
sr_4_delete40=sr_4_delete39[ ~ sr_4['TAG_LV2'].str.contains('外食族')]
sr_4_delete41=sr_4_delete40[ ~ sr_4['TAG_LV2'].str.contains('大潤發')]
sr_4_delete42=sr_4_delete41[ ~ sr_4['TAG_LV2'].str.contains('大眾休閒')]
sr_4_delete43=sr_4_delete42[ ~ sr_4['TAG_LV2'].str.contains('大阪')]
sr_4_delete44=sr_4_delete43[ ~ sr_4['TAG_LV2'].str.contains('奈良')]
sr_4_delete45= sr_4_delete44[ ~ sr_4['TAG_LV2'].str.contains('套裝行程')]
sr_4_delete46=sr_4_delete45[ ~ sr_4['TAG_LV2'].str.contains('女性')]
sr_4_delete47=sr_4_delete46[ ~ sr_4['TAG_LV2'].str.contains('女性媒體')]
sr_4_delete48=sr_4_delete47[ ~ sr_4['TAG_LV2'].str.contains('嬰幼兒')]
sr_4_delete49=sr_4_delete48[ ~ sr_4['TAG_LV2'].str.contains('孩童')]
sr_4_delete50=sr_4_delete49[ ~ sr_4['TAG_LV2'].str.contains('家庭清潔')]
sr_4_delete51=sr_4_delete50[ ~ sr_4['TAG_LV2'].str.contains('家樂福')]
sr_4_delete52=sr_4_delete51[ ~ sr_4['TAG_LV2'].str.contains('寶雅')]
sr_4_delete53=sr_4_delete52[ ~ sr_4['TAG_LV2'].str.contains('小三美日')]
sr_4_delete54=sr_4_delete53[ ~ sr_4['TAG_LV2'].str.contains('屈臣氏')]
sr_4_delete55=sr_4_delete54[ ~ sr_4['TAG_LV2'].str.contains('床')]
sr_4_delete56=sr_4_delete55[ ~ sr_4['TAG_LV2'].str.contains('康是美')]
sr_4_delete57=sr_4_delete56[ ~ sr_4['TAG_LV2'].str.contains('影視娛樂')]
sr_4_delete58=sr_4_delete57[ ~ sr_4['TAG_LV2'].str.contains('御宅族')]
sr_4_delete59=sr_4_delete58[ ~ sr_4['TAG_LV2'].str.contains('愛狗人士')]
sr_4_delete60=sr_4_delete59[ ~ sr_4['TAG_LV2'].str.contains('愛貓人士')]
sr_4_delete61=sr_4_delete60[ ~ sr_4['TAG_LV2'].str.contains('戶外休閒')]
sr_4_delete62=sr_4_delete61[ ~ sr_4['TAG_LV2'].str.contains('健身族')]
sr_4_delete63=sr_4_delete62[ ~ sr_4['TAG_LV2'].str.contains('搖滾樂')]
sr_4_delete64=sr_4_delete63[ ~ sr_4['TAG_LV2'].str.contains('攝影')]
sr_4_delete65=sr_4_delete64[ ~ sr_4['TAG_LV2'].str.contains('收納')]
sr_4_delete66=sr_4_delete65[ ~ sr_4['TAG_LV2'].str.contains('政治')]
sr_4_delete67=sr_4_delete66[ ~ sr_4['TAG_LV2'].str.contains('文化藝術')]
sr_4_delete68=sr_4_delete67[ ~ sr_4['TAG_LV2'].str.contains('日本')]
sr_4_delete69=sr_4_delete68[ ~ sr_4['TAG_LV2'].str.contains('日本料理')]
sr_4_delete70=sr_4_delete69[ ~ sr_4['TAG_LV2'].str.contains('日本節目')]
sr_4_delete71=sr_4_delete70[ ~ sr_4['TAG_LV2'].str.contains('日本音樂')]
sr_4_delete72=sr_4_delete71[ ~ sr_4['TAG_LV2'].str.contains('日系動漫')]
sr_4_delete73=sr_4_delete72[ ~ sr_4['TAG_LV2'].str.contains('日藥本舖')]
sr_4_delete74=sr_4_delete73[ ~ sr_4['TAG_LV2'].str.contains('春季')]
sr_4_delete75=sr_4_delete74[ ~ sr_4['TAG_LV2'].str.contains('時尚')]
sr_4_delete76=sr_4_delete75[ ~ sr_4['TAG_LV2'].str.contains('朋友')]
sr_4_delete77=sr_4_delete76[ ~ sr_4['TAG_LV2'].str.contains('東京')]
sr_4_delete78=sr_4_delete77[ ~ sr_4['TAG_LV2'].str.contains('東南亞')]
sr_4_delete79=sr_4_delete78[ ~ sr_4['TAG_LV2'].str.contains('格鬥與摔角')]
sr_4_delete80=sr_4_delete79[ ~ sr_4['TAG_LV2'].str.contains('棒球')]
sr_4_delete81=sr_4_delete80[ ~ sr_4['TAG_LV2'].str.contains('歐美')]
sr_4_delete82=sr_4_delete81[ ~ sr_4['TAG_LV2'].str.contains('歐美節目')]
sr_4_delete83=sr_4_delete82[ ~ sr_4['TAG_LV2'].str.contains('水上運動')]
sr_4_delete84=sr_4_delete83[ ~ sr_4['TAG_LV2'].str.contains('水族寵物')]
sr_4_delete85=sr_4_delete84[ ~ sr_4['TAG_LV2'].str.contains('沖繩')]
sr_4_delete86=sr_4_delete85[ ~ sr_4['TAG_LV2'].str.contains('沙發')]
sr_4_delete87=sr_4_delete86[ ~ sr_4['TAG_LV2'].str.contains('游泳')]
sr_4_delete88=sr_4_delete87[ ~ sr_4['TAG_LV2'].str.contains('濟州島')]
sr_4_delete89=sr_4_delete88[ ~ sr_4['TAG_LV2'].str.contains('烹飪')]
sr_4_delete90=sr_4_delete89[ ~ sr_4['TAG_LV2'].str.contains('照明')]
sr_4_delete91=sr_4_delete90[ ~ sr_4['TAG_LV2'].str.contains('男性')]
sr_4_delete92=sr_4_delete91[ ~ sr_4['TAG_LV2'].str.contains('男性媒體')]
sr_4_delete93=sr_4_delete92[ ~ sr_4['TAG_LV2'].str.contains('百貨公司')]
sr_4_delete94=sr_4_delete93[ ~ sr_4['TAG_LV2'].str.contains('節慶')]
sr_4_delete95=sr_4_delete94[ ~ sr_4['TAG_LV2'].str.contains('籃球')]
sr_4_delete96=sr_4_delete95[ ~ sr_4['TAG_LV2'].str.contains('素食')]
sr_4_delete97=sr_4_delete96[ ~ sr_4['TAG_LV2'].str.contains('綠色環保')]
sr_4_delete98=sr_4_delete97[ ~ sr_4['TAG_LV2'].str.contains('網球')]
sr_4_delete99=sr_4_delete98[ ~ sr_4['TAG_LV2'].str.contains('網路購物')]
sr_4_delete100=sr_4_delete99[ ~ sr_4['TAG_LV2'].str.contains('美式料理')]
sr_4_delete101=sr_4_delete100[ ~ sr_4['TAG_LV2'].str.contains('美系動漫')]
sr_4_delete102=sr_4_delete101[ ~ sr_4['TAG_LV2'].str.contains('老年銀髮')]
sr_4_delete103=sr_4_delete102[ ~ sr_4['TAG_LV2'].str.contains('臉部彩妝')]
sr_4_delete104=sr_4_delete103[ ~ sr_4['TAG_LV2'].str.contains('臉部清潔保養')]
sr_4_delete105=sr_4_delete104[ ~ sr_4['TAG_LV2'].str.contains('自行車')]
sr_4_delete106=sr_4_delete105[ ~ sr_4['TAG_LV2'].str.contains('航空')]
sr_4_delete107=sr_4_delete106[ ~ sr_4['TAG_LV2'].str.contains('華人影視娛樂')]
sr_4_delete108=sr_4_delete107[ ~ sr_4['TAG_LV2'].str.contains('菸酒')]
sr_4_delete109=sr_4_delete108[ ~ sr_4['TAG_LV2'].str.contains('萊爾富')]
sr_4_delete110=sr_4_delete109[ ~ sr_4['TAG_LV2'].str.contains('藥妝店')]
sr_4_delete111=sr_4_delete110[ ~ sr_4['TAG_LV2'].str.contains('行動支付')]
sr_4_delete112=sr_4_delete111[ ~ sr_4['TAG_LV2'].str.contains('行動裝置')]
sr_4_delete113=sr_4_delete112[ ~ sr_4['TAG_LV2'].str.contains('裝飾')]
sr_4_delete114=sr_4_delete113[ ~ sr_4['TAG_LV2'].str.contains('親子')]
sr_4_delete115=sr_4_delete114[ ~ sr_4['TAG_LV2'].str.contains('資訊科技')]
sr_4_delete116=sr_4_delete115[ ~ sr_4['TAG_LV2'].str.contains('賣場超市')]
sr_4_delete117=sr_4_delete116[ ~ sr_4['TAG_LV2'].str.contains('賽車')]
sr_4_delete118=sr_4_delete117[ ~ sr_4['TAG_LV2'].str.contains('超值買家')]
sr_4_delete119=sr_4_delete118[ ~ sr_4['TAG_LV2'].str.contains('超商')]
sr_4_delete114=sr_4_delete113[ ~ sr_4['TAG_LV2'].str.contains('足球')]
sr_4_delete115=sr_4_delete114[ ~ sr_4['TAG_LV2'].str.contains('跑步')]
sr_4_delete116=sr_4_delete115[ ~ sr_4['TAG_LV2'].str.contains('身體保健')]
sr_4_delete117=sr_4_delete116[ ~ sr_4['TAG_LV2'].str.contains('身體清潔保養')]
sr_4_delete118=sr_4_delete117[ ~ sr_4['TAG_LV2'].str.contains('速食')]
sr_4_delete119=sr_4_delete118[ ~ sr_4['TAG_LV2'].str.contains('遊戲-單機遊戲')]
sr_4_delete114=sr_4_delete113[ ~ sr_4['TAG_LV2'].str.contains('遊戲-手機遊戲')]
sr_4_delete115=sr_4_delete114[ ~ sr_4['TAG_LV2'].str.contains('遠傳')]
sr_4_delete116=sr_4_delete115[ ~ sr_4['TAG_LV2'].str.contains('釜山')]
sr_4_delete117=sr_4_delete116[ ~ sr_4['TAG_LV2'].str.contains('鐵公路')]
sr_4_delete118=sr_4_delete117[ ~ sr_4['TAG_LV2'].str.contains('電影')]
sr_4_delete119=sr_4_delete118[ ~ sr_4['TAG_LV2'].str.contains('電視購物')]
sr_4_delete119=sr_4_delete118[ ~ sr_4['TAG_LV2'].str.contains('韓國')]
sr_4_delete114=sr_4_delete113[ ~ sr_4['TAG_LV2'].str.contains('韓國節目')]
sr_4_delete115=sr_4_delete114[ ~ sr_4['TAG_LV2'].str.contains('跑步')]
sr_4_delete116=sr_4_delete115[ ~ sr_4['TAG_LV2'].str.contains('韓國音樂')]
sr_4_delete117=sr_4_delete116[ ~ sr_4['TAG_LV2'].str.contains('頂好')]
sr_4_delete118=sr_4_delete117[ ~ sr_4['TAG_LV2'].str.contains('頭髮清潔造型')]
sr_4_delete119=sr_4_delete118[ ~ sr_4['TAG_LV2'].str.contains('食品')]
sr_4_delete114=sr_4_delete113[ ~ sr_4['TAG_LV2'].str.contains('飲料')]
sr_4_delete115=sr_4_delete114[ ~ sr_4['TAG_LV2'].str.contains('首爾')]
sr_4_delete116=sr_4_delete115[ ~ sr_4['TAG_LV2'].str.contains('高爾夫球')]    
```

##### 依據欄位srno,TAG_LV2,UPD_DT加總 score

```python
sr_4_delete_loan= sr_4_delete116
srno=np.array(sr_4_delete_loan['srno'])
TAG_LV2=np.array(sr_4_delete_loan['TAG_LV2'])
UPD_DT=np.array(sr_4_delete_loan['UPD_DT'])
score=np.array(sr_4_delete_loan['score'])
sr_4_score_loan=sr_4_delete_loan['score'].groupby([srno,TAG_LV2,UPD_DT]).sum().to_frame('score').reset_index()
```

##### 輸出檔案

```python
sr_4_score_loan.to_csv('sr_4_loan.csv') 
```


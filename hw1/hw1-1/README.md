# README

### 語言
Python 3.6.5

### 使用套件
1. Selenium
* 原因：在yahoo finance中下載ETF歷史資料的button是動態載入的，因此使用selenium模仿人為操縱瀏覽器，並且使用chrome的headless模式背景執行。
2. pandas, ExcelWriter
3. csv

### 程式說明
主要分為etf_scrapy.py和arrange_data.py兩個檔案
* **etf_scrapy.py**：利用selenium套件去yahoo finance下載各etf的歷史資料(csv檔)
* **arrange_data.py**：將各etf的csv檔中的價格結合並且製作成新的dataframe(有額外輸出成"output.xlsx")
* **備註**：因為這次分配到的etf中大部分homepage中都沒有價格的資料，因此主要以yahoo finance中的歷史資料為主

### 流程圖
![](https://i.imgur.com/GWytoTx.png)

### 程式輸出
![](https://i.imgur.com/HxiNwWd.png)
*資料夾中有輸出的"output.xlsx"檔


### 可能遇到情況
1. 因為yahoo finance的網站有很多廣告，因此載入的時間會有點久，在etf_scrapy.py中有設定implicitly_wait為10秒鐘，若還是無法下載到歷史資料的話，可以增加implicitly_wait的時間
2. 如果想要觀察selenium的執行狀況，可以將chrome_option的headless模式註解掉
3. arrange_data.py中要讀的是csv檔，若資料是xlsx檔的話須先轉成csv檔再執行程式
4. "etf file.xlsx"中為2015年以前即存在的etf清單，可以從該excel檔去做新增etf的操作


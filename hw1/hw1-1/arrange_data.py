import pandas as pd
import csv
from pandas.core.frame import DataFrame
from pandas import ExcelWriter

# read excel file
df = pd.read_excel("Etf List.xlsx")
df = pd.DataFrame(df)

# 將etf的名字存入name_list
name_list = []
etf_num = 0
for index, row in df.iterrows():
    etf_num += 1
    name_list.append(row['Symbol'])
# print(name_list)

# open file
data_list = []
record_date = False


# 將日期存入data_list
etf_file = pd.read_csv("excel file/" + name_list[0] + ".csv")
etf_file = pd.DataFrame(etf_file)
data_list.append(["Date"])
for index, row in etf_file.iterrows():
    if row['Date'] == "2015-12-31":
        record_date = True
    if record_date == True:
        data_list[0].append(row['Date'])

# 將Adj Close存入data_list
for i in range(len(name_list)):
    start = False
    etf_file = pd.read_csv("excel file/" + name_list[i] + ".csv")
    etf_file = pd.DataFrame(etf_file)
    data_list.append([name_list[i]])
    for index, row in etf_file.iterrows():
        if row['Date'] == "2015-12-31":
            start = True
        elif row['Date'] > "2015-12-31":
            start = True
        if start == True:
            data_list[i + 1].append(row['Adj Close'])

# 將list轉為DataFrame
final_table = DataFrame(data_list)
final_table = final_table.T

print(final_table)

# 將DataFrame輸出成excel檔
writer = ExcelWriter('output.xlsx')
final_table.to_excel(writer,'Sheet1')
writer.save()

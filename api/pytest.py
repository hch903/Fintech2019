import pandas as pd
from sklearn.svm import SVC
IO = "ETF0607.xlsx"
weight_compose =  pd.read_excel(IO,2)
for i in range(len(weight_compose)):
	print(weight_compose.iloc[i][11]*1)
	if(weight_compose.iloc[i][11]):
		a = weight_compose.iloc[i][11].copy()
		a = 0
	print(a+1)
	exit()
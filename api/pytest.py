import pandas as pd
from sklearn.svm import SVC
import os
import math
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data") 
IO = os.path.join(DATA_DIR,"ETF0607.xlsx")
weight_compose =  pd.read_excel(IO,2)
d = {0:[]}
for i in range(len(weight_compose.iloc[0])):
	print(weight_compose.iloc[0][i])
	# for j in range(1,len(weight_compose.iloc[i])):
	# 	if(math.isnan(weight_compose.iloc[i][j])):
	# 		print(j)
	

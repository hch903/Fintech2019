import pandas as pd
import numpy as np
import pickle
from final_train import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
IO = 'company_up_down_news(2016_2017).xlsx'
SHEET_NUM = 6
WORD_TO_VECTOR_MODEL = "finword_to_vec.model"
WORD_VECTOR_DIM = 50
TITLE_LENGTH = 2
DOC_LENGTH = 50

def tfidf_with_dict(table,idf_dict):
	for sheet_index,company in table.items():
		for i in range(len(company)):
			for j in range(2,4):
				if company.iloc[i][j] :
					temp = []
					for index,word in enumerate(company.iloc[i][j]):
						if word not in idf_dict.keys():
							w_idf = 0
						else:
							w_idf = idf_dict[word]
						tfidf = tf(word,company.iloc[i][j])*w_idf
						word = (word,tfidf)
						if word not in temp and word[0] !=" ":
							temp.append(word)
					temp = sorted(temp,key = lambda x:x[1],reverse = True) # sort by tfidf
					temp = [w[0] for w in temp]
					if(j == 2):
						try:
							company.iloc[i][j] = temp[:TITLE_LENGTH]# each title only content TITLE_LENGTH words
						except Exception as e:
							dummy = (TITLE_LENGTH-len(temp))*[" "]
							company.iloc[i][j] = temp + dummy 
					else:
						try:
							company.iloc[i][j] = temp[:DOC_LENGTH]# each content only content DOC_LENGTH words
						except Exception as e:
							dummy = (DOC_LENGTH-len(temp))*[" "]
							company.iloc[i][j] = temp + dummy  
	return table
def test_model(sheet1,sheet2,model): #sheet1 for postive news,  sheet2 for negitave news
	x_test = []
	y_test = []
	for i in range(len(sheet1)):
		temp = sheet1.iloc[i][2]+sheet1.iloc[i][3]
		print(temp)
		exit()
		x_test.append(temp)
		y_test.append(1)
	for i in range(len(sheet2)):
		temp = sheet2.iloc[i][2]+sheet2.iloc[i][3]
		x_test.append(temp)
		y_test.append(0)
	x_test = np.array(x_test)
	y_test = np.array(y_test)
	shape =  x_test.shape
	try:
		x_test = x_test.reshape((shape[0],shape[1]*shape[2]))
	except:
		print("shape",shape)
		exit()
	predict = model.predict(x_test)
	print (accuracy_score(y_test,predict))
	return predict


def main():
	table = {}
	table[0] = preprocess(pd.read_excel(IO,0)[250:326])
	table[1] = preprocess(pd.read_excel(IO,1)[250:275])
	# table[2] = preprocess(pd.read_excel(IO,2)[400:474])
	# table[3] = preprocess(pd.read_excel(IO,3)[400:485])
	# table[4] = preprocess(pd.read_excel(IO,4)[100:164])
	# table[5] = preprocess(pd.read_excel(IO,5)[100:119])

	word_vector  = word2vec.Word2Vec.load(WORD_TO_VECTOR_MODEL)
	with open("idf_dict.json") as f:
		idf_dict = json.load(f)
		table = tfidf_with_dict(table,idf_dict)
	
	
	table = word_2_vec(table,word_vector)
	print (table[0])
	exit()
	

	with open("tsmc_model",'rb') as f:
		tsmc_model = pickle.load(f)
		tsmc_predict = test_model(table[0],table[1],tsmc_model)
	with open("foxconn_model",'rb') as f:
		foxconn_model = pickle.load(f)
		fox_predict = test_model(table[2],table[3],foxconn_model)
	with open("fpc_model",'rb') as f:
		fpc_model = pickle.load(f)
		fpc_predict = test_model(table[4],table[5],fpc_model)
	
	

	return

if __name__ == '__main__':
	main()

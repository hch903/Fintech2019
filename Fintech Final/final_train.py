import pandas as pd
import numpy as np 
import logging
import datetime
import jieba
import pickle
import json
from gensim.models import word2vec
from sklearn.svm import SVC
from math import log
logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s- %(levelname)s:%(message)s',
					datefmt='%Y%m%d%H%M%S',
					filename='finallog.txt')
# logging.info('info message')
IO = 'company_up_down_news(2016_2017).xlsx'
SHEET_NUM = 6
WORD_TO_VECTOR_MODEL = "finword_to_vec.model"
WORD_VECTOR_DIM = 50
TITLE_LENGTH = 2
DOC_LENGTH = 50
def init():
	table = dict()
	for i in range(SHEET_NUM):
		table[i] = preprocess(pd.read_excel(IO,i))
	return table
def preprocess(sheet):
	stopword_set = set()
	with open("stopwords.txt","r",encoding="utf-8") as stopwords:
		for stopword in stopwords:
			stopword_set.add(stopword.strip('\n'))
	for i in range(len(sheet)):
		sheet.iloc[i][2] = segment(sheet.iloc[i][2],stopword_set)
		sheet.iloc[i][3] = segment(sheet.iloc[i][3],stopword_set)
	return sheet
def segment(sentence,stopword_set):
	try:
		sentence = sentence.strip('\n')
		for word in stopword_set:
			sentence.strip(word)
		words = jieba.cut(sentence,cut_all = True)
		#words = [word+" " for word in words ]
		return list(words)
	except:
		logging.warning("segment fail")
		return []
def train_word_2_vec(table):
	train_sent = ""
	for key,value in table.items():
		for i in range(len(value)):
			for word in value.iloc[i][2]:
				train_sent += word # title
			for word in value.iloc[i][3]:
				train_sent += word # content
			train_sent +="\n"
	logging.info(train_sent)
	model = word2vec.Word2Vec(train_sent,size = WORD_VECTOR_DIM ,window =2,min_count = 5)
	model.save(WORD_TO_VECTOR_MODEL)
	return  model.wv
def word_2_vec(table,word_vector):
	for sheet_index,company in table.items():
		for i in range(len(company)):
			for j in range(2,4):
				if company.iloc[i][j]:
					temp = []
					for index,word in enumerate(company.iloc[i][j]):
						if word in word_vector.vocab and word!=" " :
							temp.append(word_vector.wv[word])
						else:
							temp.append([0]*WORD_VECTOR_DIM)
					company.iloc[i][j] = temp 
		table[sheet_index] = company
	return table
def tfidf(table,gen_idf_dict = False):
	idf_dict = {}
	for sheet_index,company in table.items():
		for i in range(len(company)):
			for j in range(2,4):
				if company.iloc[i][j] :
					temp = []
					for index,word in enumerate(company.iloc[i][j]):
						w_idf = idf(word,company.iloc[:,j])
						tfidf = tf(word,company.iloc[i][j])*w_idf
						if word not in idf_dict.keys() and word !=" ":
							idf_dict[word] = w_idf
						word = (word,tfidf)
						if word not in temp and word[0] !=" ":
							temp.append(word)
						
					temp = sorted(temp,key = lambda x:x[1],reverse = True) # sort by tfidf
					temp = [w[0] for w in temp]
					if(j == 2):
						try:
							company.iloc[i][j] = temp[:TITLE_LENGTH]# each title only content title_length words
						except Exception as e:
							dummy = (TITLE_LENGTH-len(temp))*[" "]
							company.iloc[i][j] = temp + dummy 
					else:
						try:
							company.iloc[i][j] = temp[:DOC_LENGTH]# each content only content DOC_LENGTH words
						except Exception as e:
							dummy = (DOC_LENGTH-len(temp))*[" "]
							company.iloc[i][j] = temp + dummy
	if gen_idf_dict:  
		with open("idf_dict.json","w",encoding = 'utf8') as fp:
			json.dump(idf_dict,fp)

	return table
def tf(word,doc):
	count = 0
	size = len(doc)
	for w in doc:
		if(word == w):
			count+=1
	return float(count)/size
def idf(word,docs):
	n = len(docs)
	count = 0
	for i in range(n):
		doc = docs.iloc[i]
		if doc :
			for w in doc[0]:
				if(word == w):
					count+=1
					break
	return log(n/(count+1),10)
def build_model(sheet1,sheet2,model_name): #sheet1 for postive news,  sheet2 for negitave news
	x_train = []
	y_train = []
	for i in range(len(sheet1)):
		temp = sheet1.iloc[i][2]+sheet1.iloc[i][3]
		x_train.append(temp)
		y_train.append(1)
	for i in range(len(sheet2)):
		temp = sheet2.iloc[i][2]+sheet2.iloc[i][3]
		x_train.append(temp)
		y_train.append(0)
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	shape =  x_train.shape
	try:
		x_train = x_train.reshape((shape[0],shape[1]*shape[2]))
	except:
		print("shape",shape)
		print("x_train",x_train)
		exit()
	clf = SVC(gamma='auto',kernel='rbf',random_state = 10)
	clf.fit(x_train,y_train.ravel())
	with open(model_name,'wb') as f:
		pickle.dump(clf,f)
	return
		
def main():
	table = init()
	# table = {}
	# table[0] = preprocess(pd.read_excel(IO,0)[:250])
	# table[1] = preprocess(pd.read_excel(IO,1)[:250])
	# table[2] = preprocess(pd.read_excel(IO,2)[:400])
	# table[3] = preprocess(pd.read_excel(IO,3)[:400])
	# table[4] = preprocess(pd.read_excel(IO,4)[:100])
	# table[5] = preprocess(pd.read_excel(IO,5)[:100])

	try:
	 	word_vector  = word2vec.Word2Vec.load(WORD_TO_VECTOR_MODEL).wv
	except Exception as e:
		print(e)
		print("train word2vec by ourself..")
		word_vector = train_word_2_vec(table)
	table = tfidf(table,True)
	table = word_2_vec(table,word_vector)
	build_model(table[0],table[1],"tsmc_model")
	print("tsmc_model down")
	build_model(table[2],table[3],"foxconn_model")
	print("foxconn_model down")
	build_model(table[4],table[5],"fpc_model")
	print("fpc_model down")
	return
if __name__ == '__main__':
	main()

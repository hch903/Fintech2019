import logging
import jieba
import io
import json
import pickle
import os
import numpy as np
import pandas as pd
import math
import copy
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from gensim.models import word2vec
from sklearn.svm import SVC

from django.conf import settings
logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s- %(levelname)s:%(message)s',
					datefmt='%Y%m%d%H%M%S',
					filename='apilog.txt')


# Create your views here.
WORD_TO_VECTOR_MODEL = os.path.join(settings.DATA_DIR,"finword_to_vec.model")
IO = os.path.join(settings.DATA_DIR,"ETF0607.xlsx")
STOPWORD = os.path.join(settings.DATA_DIR,"stopwords.txt") 
WORD_VECTOR_DIM = 20
TITLE_LENGTH = 5
DOC_LENGTH = 25
THRESHOLD = 0.65
try:
	word_vector  = word2vec.Word2Vec.load(WORD_TO_VECTOR_MODEL).wv
	weight_compose =  pd.read_excel(IO,2)
except Exception as e:
	print(e)
def preprocess(sentence):
	stopword_set = set()
	with open(STOPWORD,"r",encoding="utf-8") as stopwords:
		for stopword in stopwords:
			stopword_set.add(stopword.strip('\n'))
	
	sentence = segment(sentence,stopword_set)
	return sentence

def segment(sentence,stopword_set):
	try:
		sentence = sentence.strip('\n')
		for word in stopword_set:
			sentence.strip(word)
		words = jieba.cut(sentence,cut_all = True)
		#words = [word+" " for word in words ]
		return list(words)
	except Exception as e:
		logging.warning("segment fail",exc_info=True)
		return []
def model_predict(title,content,model): 
	x_test = []
	temp = np.concatenate((title,content))
	x_test.append(temp)
	x_test = np.array(x_test)
	shape =  x_test.shape
	try:
		x_test = x_test.reshape((shape[0],shape[1]*shape[2]))
	except:
		print("shape",shape)
		exit()
	vote = model.predict(x_test)
	proba = model.predict_proba(x_test)
	#proba = 1/(1+math.exp(-1*predict))
	proba = max(proba[0])
	#print(type(proba))
	#print(proba[0][0])
	if proba <= THRESHOLD:
		proba = "not affected"
	return [proba,vote[0]]
def tfidf(title,content,idf_dict):
	t_title = []
	t_content = []
	for index,word in enumerate(title):
		w_idf = idf_dict.get(word,0)
		tfidf = tf(word,title)*w_idf
		word = (word,tfidf)
		if word not in t_title and word[0] !=" ":
			t_title.append(word)	
	t_title = sorted(t_title,key = lambda x:x[1],reverse = True) # sort by tfidf
	#t_title = [w[0] for w in t_title]
	for index,word in enumerate(content):
		w_idf = idf_dict.get(word,0)
		tfidf = tf(word,content)*w_idf
		word = (word,tfidf)
		if word not in t_content and word[0] != " ":
			t_content.append(word)
	t_content = sorted(t_content,key = lambda x:x[1],reverse = True)
	#t_content = [w[0] for w in t_content]
	if(len(t_title)>= TITLE_LENGTH):
		t_title = [word_2_vec(w[0])*w[1] for w in t_title]
		title = t_title[:TITLE_LENGTH]# each title only content title_length words
	else:
		dummy = (TITLE_LENGTH-len(t_title))*[(" ",0)]
		t_title = t_title+dummy
		t_title = [word_2_vec(w[0])*w[1] for w in t_title]
		title = np.array(t_title)
		
	if(len(t_content)>=DOC_LENGTH):
		t_content = [word_2_vec(w[0])*w[1] for w in t_content]
		content = t_content[:DOC_LENGTH]# each content only content DOC_LENGTH words
	else:
		dummy = (TITLE_LENGTH-len(t_content))*[(" ",0)]
		t_content = t_content+dummy
		t_content = [word_2_vec(w[0])*w[1] for w in t_content]
		content = np.array(t_content)
	return title,content
def tf(word,doc):
	count = 0
	size = len(doc)
	for w in doc:
		if(word == w):
			count+=1
	return float(count)/size

def word_2_vec(word):
	if word in word_vector.vocab and word!=" " :
		return np.array(word_vector.wv[word])
	else:
		return np.array([0]*WORD_VECTOR_DIM)
class score(APIView):
	def post(self,request):
		


		title = preprocess(request.data.get("title"))
		content = preprocess(request.data.get("content"))
		score_l = {}
		ans = {}
		if title == []or content == []:
			return Response({"msg":"title or content can't be null"})
		for company in range(11):
			IDF_DIC = os.path.join(settings.DATA_DIR,"idf_dict_%s_company.json"%str(company))
			COMPANY_MODEL = os.path.join(settings.DATA_DIR,"company%s_model"%(str(company)))
			with io.open(IDF_DIC,"r",encoding = 'utf8') as fp:
				idf_dict = json.load(fp)
			with open(COMPANY_MODEL,'rb') as f:
				model = pickle.load(f)
			t_title,t_content = tfidf(title.copy(),content.copy(),idf_dict)
			#t_title = word_2_vec(t_title,word_vector)
			#t_content = word_2_vec(t_content,word_vector)
			score_l[company] = model_predict(t_title,t_content,model)

		

		for i in range(len(weight_compose)):
			temp  = copy.deepcopy(score_l)

			for comp in list(temp.keys()):
				if(temp[comp][0] == "not affected"):
					del temp[comp]
			for j in range(1,len(weight_compose.iloc[i])):
				if j in temp.keys():
					#print(temp[j])
					if(math.isnan(weight_compose.iloc[i][j])):
						temp[j].append(0)
					else:
						temp[j].append(weight_compose.iloc[i][j])
			ans[weight_compose.iloc[i][0]] = temp

			
		
		


		return Response({"msg":ans})
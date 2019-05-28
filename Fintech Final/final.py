import pandas as pd 
import logging
import datetime
import jieba
from gensim.models import word2vec
logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s- %(levelname)s:%(message)s',
					datefmt='%Y%m%d%H%M%S',
					filename='finallog.txt')
# logging.info('info message')
IO = 'company_up_down_news(2016_2017).xlsx'
SHEET_NUM = 6
WORD_TO_VECTOR_MODEL = "finword_to_vec.model"
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
		words = jieba.cut(sentence,cut_all = False)
		words = [word+" " for word in words if word not in stopword_set]
		return words
	except:
		logging.warning(sentence)
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
	model = word2vec.Word2Vec(train_sent,size = 250 ,window =2)
	model.save(WORD_TO_VECTOR_MODEL)
	return  model


			
def main():
	table = dict()
	table["tsmc_up"] = preprocess(pd.read_excel(IO,0))
	table["tsmc_down"] = preprocess(pd.read_excel(IO,1))
	table["foxconn_up"] = preprocess(pd.read_excel(IO,2))
	table["foxconn_down"] = preprocess(pd.read_excel(IO,3))
	table["fpc_up"] = preprocess(pd.read_excel(IO,4))
	table["fpc_down"] = preprocess(pd.read_excel(IO,5))
	try:
	 	word_model  = word2vec.Word2Vec.load(WORD_TO_VECTOR_MODEL)
	except Exception as e:
		print(e)
		print("train word2vec by ourself..")
	finally:
		word_model = train_word_2_vec(table)

	return
if __name__ == '__main__':
	main()
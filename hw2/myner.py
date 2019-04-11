import json
import gensim.downloader as api
import gensim
import spacy
import pandas as pd
import re
import csv
from collections import defaultdict as dd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel
pd.set_option("display.width",None)
stw=stopwords.words('english')
stw.extend(['from', 'subject', 're', 'edu', 'use','etfs','us','esg','qqq','also','pm','billion','million','etf','market','fund','year','month','think','one'])
CLASS_NUM = 3
def preprocess(sentence):
	sentence = sentence.replace(",","")
	sentence = sentence.replace(".","")
	tokenizer =RegexpTokenizer(r'\w+')
	sentence=sentence.lower()
	docs = tokenizer.tokenize(sentence)   # token
	docs = [doc for doc in docs if  doc not in stw] # remove stopword
	
	return docs
def create_data(file_name):
	description_data = []
	with open(file_name , 'r' ) as f:
		raw_dic = json.load(f)
	for page,info in raw_dic.items():
		description_data.append(preprocess(info["topic"] +' ' + info['parag']))  
	return description_data
def make_bigrams(texts,bigram_mod):
	return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts,trigram_mod,bigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts,nlp ,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
	texts_out = []
	for sent in texts:
		doc = nlp(" ".join(sent)) 
		texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
	return texts_out
def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()

    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)



def main():
	data_words = create_data("output.json")
	bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
	trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
	
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	trigram_mod = gensim.models.phrases.Phraser(trigram)
	
	data_words_bigrams = make_bigrams(data_words,bigram_mod)
	data_words_trigrams = make_trigrams(data_words,trigram_mod,bigram_mod)
	
	nlp = spacy.load('en', disable=['parser', 'ner'])

	data_lemmatized = lemmatization(data_words_trigrams, nlp,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
	
	id2word = corpora.Dictionary(data_lemmatized)

	corpus = [id2word.doc2bow(text) for text in data_words_trigrams]
	#print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

	lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,id2word = id2word,num_topics = CLASS_NUM,
												random_state = 100,update_every = 1,chunksize = 100,passes = 25,
												alpha = 'auto',per_word_topics = True)
	#print("lda_model.topic\n",lda_model.print_topics())
	null_topics = lda_model.print_topics()
	#print(type(a),a)
	
	classify_dict = {}
	for row in null_topics:
		self_type,key_word_combind = row[0],row[1]
		key_word_list = ''.join(c for c in key_word_combind if (c.isalnum() or c.isspace()) and not c.isdigit()).split()
		for key_word in key_word_list:
			classify_dict.setdefault(key_word,self_type)
	temp_cla_dict = dd(list)
	for w ,self_type in classify_dict.items():
		temp_cla_dict[self_type].append(w)
	classify_dict = temp_cla_dict

	print("classify word",classify_dict)
	with open("output.json" , 'r' ) as f:
		output = json.load(f) 
		tdm = []
		co_tdm = []
		key_word_list = []
		for i,words in classify_dict.items():
			key_word_list += words
		
		
		for index,info in output.items():
			temp_tdm = [0,0,0]
			temp_co_tdm = [0]*len(key_word_list)
			words = preprocess(info["topic"] +' ' + info['parag'])
			words = [(w,-1) for w in words]
			for w in words:
				for key,val in classify_dict.items():
					if w[0] in val:
						temp_tdm[key] = temp_tdm[key]+1
						w = (w[0],key)
				for key, val in enumerate(key_word_list):
					if w[0] == val:
						temp_co_tdm[key] = temp_co_tdm[key]+1
			tdm.append(temp_tdm)
			co_tdm.append(temp_co_tdm)
			info['parag'] = words
	with open("result.json",'w') as fp:
		json.dump(output,fp)
	print("tdm",tdm)
	print("\nco_tdm",co_tdm)
	total = len(co_tdm)
	
	fix_matrix = [0]*len(key_word_list)
	for row in co_tdm:
		for key,val in enumerate(row):
			fix_matrix[key] = fix_matrix[key] + val
	co_ocr_matrix = []
	for i in range(len(key_word_list)):
		temp = [0]*len(key_word_list)
		temp[i] = 1
		co_ocr_matrix.append(temp)

	for i in range(len(key_word_list)-1):
		for row in co_tdm:
			if(row[i]>0 and row[i+1]>0):
				co_ocr_matrix[i][i+1] = co_ocr_matrix[i][i+1] + 1/fix_matrix[i]
				co_ocr_matrix[i+1][i] = co_ocr_matrix[i+1][i] + 1/fix_matrix[i+1]
	print("key word appear times",fix_matrix)
	print("co-occurence-matrix",co_ocr_matrix)
	for key,row in enumerate(co_ocr_matrix):
		row.insert(0,key_word_list[key])
	with open("co_ocr_matrix.csv","w",newline = '') as f:
		writer = csv.writer(f)
		writer.writerow([""]+key_word_list)
		writer.writerows(co_ocr_matrix)

				






	#print(type(a[0][1]))
	#doc_lda = lda_model[corpus]
	#print('\nPerplexity: ', lda_model.log_perplexity(corpus))
	#coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
	#coherence_lda = coherence_model_lda.get_coherence()
	#print('\nCoherence Score: ', coherence_lda)
	#df_topic_sents_keywords = format_topics_sentences(lda_model,corpus, data_lemmatized)

# Format
	#df_dominant_topic = df_topic_sents_keywords.reset_index()
	#df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
	#print(df_dominant_topic.head(10))




if __name__ == '__main__':
	main()




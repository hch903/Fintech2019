import json
import gensim.downloader as api
import gensim
import spacy
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel
pd.set_option("display.width",None)
stw=stopwords.words('english')
stw.extend(['from', 'subject', 're', 'edu', 'use','etfs','us','esg','qqq','also','pm','billion','million'])
CLASS_NUM = 10
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
	
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	
	data_words_bigrams = make_bigrams(data_words,bigram_mod)
	
	nlp = spacy.load('en', disable=['parser', 'ner'])

	data_lemmatized = lemmatization(data_words_bigrams, nlp,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
	
	id2word = corpora.Dictionary(data_lemmatized)

	corpus = [id2word.doc2bow(text) for text in data_words_bigrams]
	#print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

	lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,id2word = id2word,num_topics = CLASS_NUM,
												random_state = 100,update_every = 1,chunksize = 100,passes = 10,
												alpha = 'auto',per_word_topics = True)
	print("lda_model.topic\n",lda_model.print_topics())
	doc_lda = lda_model[corpus]
	print('\nPerplexity: ', lda_model.log_perplexity(corpus))
	coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	print('\nCoherence Score: ', coherence_lda)
	df_topic_sents_keywords = format_topics_sentences(lda_model,corpus, data_lemmatized)

# Format
	df_dominant_topic = df_topic_sents_keywords.reset_index()
	df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
	print(df_dominant_topic.head(10))

	# tfidf = models.TfidfModel(corpus)
	# corpus_tfidf = tfidf[corpus]
	# lda = models.LdaModel(corpus_tfidf,id2word = dic,num_topic = CLASS_NUM,alpha = 1)
	# corpus_lda = lda[corpus_tfidf]
	# print(lda)
if __name__ == '__main__':
	main()



